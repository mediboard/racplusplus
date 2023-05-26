#include <array>
#include <unordered_set>
#include <tuple>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <set>
#include <map>
#include <mutex>
#include <iostream>
#include <thread>
#include <algorithm>
#define EIGEN_DONT_PARALLELIZE
#include <Eigen/Dense>
#include <random>


const double MIN_DISTANCE = .035;
const int NO_POINTS = 10000;

int NO_COSINE_CALLS = 0;
const int NO_CORES = 8;


// Store update neighbor times
std::vector<long> UPDATE_NEIGHBOR_DURATIONS;

// Store update NN times
std::vector<long> UPDATE_NN_DURATIONS;

// Store the durations of each call to cosine
std::vector<long> COSINE_DURATIONS;
std::vector<long> INDICES_DURATIONS;
std::vector<long> MERGE_DURATIONS;
std::vector<long> MISC_MERGE_DURATIONS;

std::array<std::array<int, NO_POINTS>, NO_CORES> MERGING_ARRAYS;

// Function to generate a matrix filled with random numbers.
Eigen::MatrixXd generateRandomMatrix(int rows, int cols, int seed) {
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distribution(0.0,1.0);

    Eigen::MatrixXd mat(rows, cols);
    for(int i=0; i<mat.rows(); ++i) {
        for(int j=0; j<mat.cols(); ++j) {
            mat(i, j) = distribution(generator);
        }
    }

    return mat;
}

class Cluster {
public:
    int id;
    bool will_merge;
    int nn;
    std::vector<int> neighbors;
    std::vector<int> indices;
    std::unordered_map<int, float> dissimilarities;
    std::vector<std::tuple<int, int, float> > neighbors_needing_updates;

    Cluster(int id)
        : id(id), will_merge(false) {
            std::vector<int> indices;
            indices.push_back(id);
            this->indices = indices;

            this->nn = -1;
        }


    void update_nn() {
        if (neighbors.size() == 0) {
            nn = -1;
            return;
        }

        float min = 1;
        int nn = -1;
        for (int neighbor : this->neighbors) {
            float dissimilarity = this->dissimilarities[neighbor];
            if (dissimilarity < min) {
                min = dissimilarity;
                nn = neighbor;
            }
        }

        this->nn = nn;
    }
};

Eigen::MatrixXd pairwise_cosine(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b) {
    // start time
    // auto start = std::chrono::high_resolution_clock::now();
    // Eigen::MatrixXd norm_a = array_a.colwise().normalized();
    // Eigen::MatrixXd norm_b = array_b.colwise().normalized();
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    // MISC_MERGE_DURATIONS.push_back(duration);

    return Eigen::MatrixXd::Ones(array_a.cols(), array_b.cols()) - (array_a.transpose() * array_b);
}

float calculate_weighted_dissimilarity(Eigen::MatrixXd points_a, Eigen::MatrixXd points_b) {
    Eigen::MatrixXd dissimilarity_matrix = pairwise_cosine(points_a, points_b);

    return static_cast<float>(dissimilarity_matrix.mean());
}

void merge_cluster(
    std::pair<int, int>& merge,
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster*>& clusters,
    std::array<int, NO_POINTS>& merging_array) {

    Cluster* main_cluster = clusters[merge.first];
    Cluster* secondary_cluster = clusters[merge.second];

    // Start timer

    // Combine current indices
    std::vector<int> new_indices;
    new_indices.reserve(main_cluster->indices.size() + secondary_cluster->indices.size());
    new_indices.insert(new_indices.end(), main_cluster->indices.begin(), main_cluster->indices.end());
    new_indices.insert(new_indices.end(), secondary_cluster->indices.begin(), secondary_cluster->indices.end());

    std::vector<int> new_neighbors;

    std::unordered_map<int, float> new_dissimilarities;
    // new_dissimilarities.reserve(main_cluster->dissimilarities.size() + secondary_cluster->dissimilarities.size());

    std::vector<int> static_neighbors;
    static_neighbors.reserve(main_cluster->neighbors.size() + secondary_cluster->neighbors.size());

    std::vector<int> merging_neighbors;
    merging_neighbors.reserve(main_cluster->neighbors.size() + secondary_cluster->neighbors.size());

    for (auto& id : main_cluster->neighbors) {
        if (id != main_cluster->id && id != secondary_cluster->id) {
            int smallest_id = id < clusters[id]->nn ? id : clusters[id]->nn;
            if (clusters[id]->will_merge) {
                if (merging_array[smallest_id] != 1) {
                    merging_array[smallest_id] = 1;
                    merging_neighbors.push_back(smallest_id);
                }
            } else {
                merging_array[id] = 1;
                static_neighbors.push_back(id);
            }
        }
    }

    for (auto& id : secondary_cluster->neighbors) {
        if (id != main_cluster->id && id != secondary_cluster->id) {
            int smallest_id = id < clusters[id]->nn ? id : clusters[id]->nn;

            if (clusters[id]->will_merge) {
                if (merging_array[smallest_id] != 1) {
                    merging_array[smallest_id] = 1;
                    merging_neighbors.push_back(smallest_id);
                }
            } else {
                if (merging_array[id] != 1) {
                    merging_array[id] = 1;
                    static_neighbors.push_back(id);
                }
            }
        }
    }

    int no_static_indices = 0;
    int no_merging_indices = 0;

    std::vector<std::pair<int, int> > static_sizes;
    static_sizes.reserve(static_neighbors.size());

    std::vector<std::pair<int, int> > merging_sizes;
    merging_sizes.reserve(merging_neighbors.size());

    std::vector<int> total_indices;
    for (int id : static_neighbors) {
        total_indices.insert(total_indices.end(), clusters[id]->indices.begin(), clusters[id]->indices.end());
        static_sizes.push_back(std::make_pair(id, clusters[id]->indices.size()));
    }
    no_static_indices = total_indices.size();

    for (int id : merging_neighbors) {
        std::vector<int> nn_indices = clusters[clusters[id]->nn]->indices;
        total_indices.insert(total_indices.end(), clusters[id]->indices.begin(), clusters[id]->indices.end());
        total_indices.insert(total_indices.end(), nn_indices.begin(), nn_indices.end());

        merging_sizes.push_back(std::make_pair(id, clusters[id]->indices.size() + nn_indices.size()));
    }
    no_merging_indices = total_indices.size() - no_static_indices;

    // auto start = std::chrono::high_resolution_clock::now();
    // New array for neighbor indices
    Eigen::MatrixXd neighbor_indices = base_arr(Eigen::all, total_indices);

    // New array for new indices
    Eigen::MatrixXd new_arr = base_arr(Eigen::all, new_indices);

    // end timer
    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    // MISC_MERGE_DURATIONS.push_back(duration);

    // Calculate dissimilarities - no batching for now
    Eigen::MatrixXd dissimilarity_matrix = pairwise_cosine(new_arr, neighbor_indices);

    // Columnwise mean
    Eigen::VectorXd dissimilarities = dissimilarity_matrix.colwise().mean();
    new_neighbors.reserve(static_sizes.size() + merging_sizes.size());

    std::vector<std::tuple<int, int, float> > needs_update;
    // Update static dismilarities
    int true_i = 0;
    for (int i=0; i<static_sizes.size(); i++) {
        int id = static_sizes[i].first;
        int size = static_sizes[i].second;

        merging_array[id] = 0; // Reset merging array

        float dissimilarity = dissimilarities.segment(true_i, size).mean();
        if (dissimilarity <= MIN_DISTANCE) {
            new_dissimilarities[id] = dissimilarity;
            new_neighbors.push_back(id);
        }
        needs_update.push_back(std::make_tuple(main_cluster->id, id, dissimilarity));

        true_i += size;
    }

    // Update merging dissimilarities
    for (int i=0; i<merging_sizes.size(); i++) {
        int id = merging_sizes[i].first;
        int size = merging_sizes[i].second;

        merging_array[id] = 0; // Reset merging array

        float dissimilarity = dissimilarities.segment(true_i, size).mean();
        if (dissimilarity <= MIN_DISTANCE) {
            new_neighbors.push_back(id);
            new_dissimilarities[id] = dissimilarity;
        }

        true_i += size;
    }

    main_cluster->neighbors = new_neighbors;
    main_cluster->dissimilarities = new_dissimilarities;

    main_cluster->neighbors_needing_updates = needs_update;
}

void merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster*>& clusters,
    std::array<int, NO_POINTS>& merging_array) {
    for (std::pair<int, int> merge : merges) {
        merge_cluster(merge, base_arr, clusters, merging_array);
    }
}

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges, 
    Eigen::MatrixXd& base_arr, 
    std::vector<Cluster*>& clusters,
    int no_threads) {

    std::vector<std::thread> threads;

    std::vector<std::vector<std::pair<int, int>>> merge_chunks(no_threads);

    int chunk_size = merges.size() / no_threads;
    int remainder = merges.size() % no_threads; 

    int start = 0, end = 0;
    for (int i = 0; i < no_threads; i++) {
        end = start + chunk_size;
        if (i < remainder) { // distribute the remainder among the first "remainder" chunks
            end++;
        }

        // Create chunks by using the range constructor of std::vector
        if (end <= merges.size()) {
            merge_chunks[i] = std::vector<std::pair<int, int>>(merges.begin() + start, merges.begin() + end);
        }

        start = end;
    }

    for (int i=0; i<no_threads; i++) {
        std::thread merge_thread = std::thread(
            merge_clusters,
            std::ref(merge_chunks[i]),
            std::ref(base_arr),
            std::ref(clusters),
            std::ref(MERGING_ARRAYS[i]));

        threads.push_back(std::move(merge_thread));
    }

    for (int i=0; i<no_threads; i++) {
        threads[i].join();
    }
}

std::array<int, NO_POINTS> UPDATE_NEIGHBORS;
void update_cluster_neighbors(std::pair<int, std::vector<std::pair<int, float> > >& update_chunk, std::vector<Cluster*>& clusters) {
    Cluster* other_cluster = clusters[update_chunk.first];

    std::vector<int> new_neighbors;
    std::vector<int> all_looped_neighbors;
    for (int i=0; i<update_chunk.second.size(); i++) {
        int neighbor_id = update_chunk.second[i].first;
        int neighbor_nn_id = clusters[neighbor_id]->nn;
        float dissimilarity = update_chunk.second[i].second;

        UPDATE_NEIGHBORS[neighbor_id] = 1;
        UPDATE_NEIGHBORS[neighbor_nn_id] = -1;

        if (dissimilarity <= MIN_DISTANCE) {
            new_neighbors.push_back(neighbor_id);
            other_cluster->dissimilarities[neighbor_id] = dissimilarity;
        } 

        all_looped_neighbors.push_back(neighbor_id);
        all_looped_neighbors.push_back(neighbor_nn_id);
    }

    for (int i=0; i<other_cluster->neighbors.size(); i++) {
        int neighbor_id = other_cluster->neighbors[i];
        if (UPDATE_NEIGHBORS[neighbor_id] == 0) {
            new_neighbors.push_back(neighbor_id);
        }
        all_looped_neighbors.push_back(neighbor_id);
    }

    // Clear array and find NN
    for (int i=0; i<all_looped_neighbors.size(); i++) {
        UPDATE_NEIGHBORS[all_looped_neighbors[i]] = 0;
    }

    other_cluster->neighbors = new_neighbors;
}

// void update_cluster_neighbors_p(std::vector<std::pair<int, int> >& updates, std::vector<Cluster*>& clusters) {
//     for (std::pair<int, int> update_pair : updates) {
//         update_cluster_neighbors(update_pair, clusters);
//     }
// }   

// void parallel_update_clusters(std::vector<std::pair<int, int> >& updates, std::vector<Cluster*>& clusters, int no_threads) {
//     std::vector<std::thread> threads;

//     std::vector<std::vector<std::pair<int, int>>> update_chunks(no_threads);

//     int chunk_size = updates.size() / no_threads;
//     int remainder = updates.size() % no_threads; 

//     int start = 0, end = 0;
//     for (int i = 0; i < no_threads; i++) {
//         end = start + chunk_size;
//         if (i < remainder) { // distribute the remainder among the first "remainder" chunks
//             end++;
//         }

//         if (end <= updates.size()) {
//             update_chunks[i] = std::vector<std::pair<int, int>>(updates.begin() + start, updates.begin() + end);
//         }
//         start = end;
//     }

//     for (int i=0; i<no_threads; i++) {
//         std::thread update_thread = std::thread(update_cluster_neighbors_p, std::ref(update_chunks[i]), std::ref(clusters));
//         threads.push_back(std::move(update_thread));
//     }

//     for (int i=0; i<no_threads; i++) {
//         threads[i].join();
//     }
// }

std::array<std::pair<int, std::vector<std::pair<int, float> > >, NO_POINTS> SORT_NEIGHBOR;

void update_cluster_dissimilarities(std::vector<std::pair<int, int> >& merges, Eigen::MatrixXd& base_arr, std::vector<Cluster*>& clusters) {
    auto start = std::chrono::high_resolution_clock::now();
    // if (merges.size() / NO_CORES > 10) {
    parallel_merge_clusters(merges, base_arr, clusters, NO_CORES);
    // } else {
    //     for (std::pair<int, int> merge : merges) {
    //         merge_cluster(merge, base_arr, clusters, MERGING_ARRAYS[0]);
    //     }
    // }
    auto end = std::chrono::high_resolution_clock::now();
    MERGE_DURATIONS.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    for (const auto& merge : merges) {
        int main = merge.first;
        int secondary = merge.second;

        clusters[main]->indices.insert(clusters[main]->indices.end(), clusters[secondary]->indices.begin(), clusters[secondary]->indices.end());

        for (int i=0; i < clusters[main]->neighbors_needing_updates.size(); i++) {
            int neighbor_idx = std::get<1>(clusters[main]->neighbors_needing_updates[i]);
            float dissimilarity = std::get<2>(clusters[main]->neighbors_needing_updates[i]);

            SORT_NEIGHBOR[neighbor_idx].first = neighbor_idx;
            SORT_NEIGHBOR[neighbor_idx].second.push_back(std::make_pair(main, dissimilarity));
        }
    }

    // This can be optimized by a vector
    start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<SORT_NEIGHBOR.size(); i++) {
        if (SORT_NEIGHBOR[i].second.size() > 0) {
            update_cluster_neighbors(SORT_NEIGHBOR[i], clusters);
            SORT_NEIGHBOR[i].second.clear();
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    UPDATE_NEIGHBOR_DURATIONS.push_back(duration.count()); 
}

void update_cluster_nn(std::vector<Cluster*>& clusters) {
    for (Cluster* cluster : clusters) {
        if (cluster == nullptr) {
            continue;
        }

        if (cluster->will_merge || (cluster->nn != -1 and clusters[cluster->nn] != nullptr and clusters[cluster->nn]->will_merge)) {
            cluster->update_nn();
        }
    }
}

void calculate_initial_disimilarities(Eigen::MatrixXd& base_arr, std::vector<Cluster*>& clusters, float min_distance) {
    // Compute the batch size
    int batchSize = 1000;  // Adjust this to fit in memory

    for (int batchStart = 0; batchStart < clusters.size(); batchStart += batchSize) {
        int batchEnd = std::min(batchStart + batchSize, static_cast<int>(clusters.size()));

        // Create a block view of the clusters in the current batch
        Eigen::MatrixXd batch = base_arr.block(0, clusters[batchStart]->indices[0], base_arr.rows(), clusters[batchEnd - 1]->indices[0] - clusters[batchStart]->indices[0] + 1);

        // Compute the pairwise cosine dissimilarity
        Eigen::MatrixXd distance_mat = pairwise_cosine(base_arr, batch).array();

        for (int i = batchStart; i < batchEnd; ++i) {
            Cluster* cluster = clusters[i];
            Eigen::VectorXd distance_vec = distance_mat.col(i - batchStart);

            std::vector<int> neighbors;
            std::unordered_map<int, float> dissimilarities;
            for (int j = 0; j < distance_vec.size(); ++j) {
                if (distance_vec[j] <= min_distance && j != cluster->id) {
                    neighbors.push_back(j);
                    dissimilarities[j] = distance_vec[j];
                }
            }

            if (!neighbors.empty()) {
                cluster->neighbors = neighbors;
                cluster->dissimilarities = dissimilarities;

                distance_vec[cluster->id] = std::numeric_limits<float>::max(); // Masking
                auto min_position = std::min_element(distance_vec.data(), distance_vec.data() + distance_vec.size());

                int nearest_neighbor = std::distance(distance_vec.data(), min_position);
                cluster->nn = nearest_neighbor;
            }
        }
    }
}

std::vector<std::pair<int, int> > find_reciprocal_nn(std::vector<Cluster*>& clusters) {
    std::vector<std::pair<int, int> > reciprocal_nn;

    for (Cluster* cluster : clusters) {
        if (cluster == nullptr) {
            continue;
        }

        if (cluster->nn != -1 && clusters[cluster->nn] != nullptr) {
            cluster->will_merge = (clusters[cluster->nn]->nn == cluster->id);
        }

        if (cluster -> will_merge && cluster->id < cluster->nn) {
            reciprocal_nn.push_back(std::make_pair(cluster->id, cluster->nn));
        }
    }

    return reciprocal_nn;
}

void remove_secondary_clusters(std::vector<std::pair<int, int> >& merges, std::vector<Cluster*>& clusters) {
    for (const auto& merge : merges) {
        int secondary_id = merge.second;
        clusters[secondary_id] = nullptr;
    }
}

void RAC_r(Eigen::MatrixXd& base_arr, std::vector<Cluster*>& clusters) {
    std::vector<std::pair<int, int> > merges = find_reciprocal_nn(clusters);
    if (merges.size() == 0) {
        return;
    }

    update_cluster_dissimilarities(merges, base_arr, clusters);

    auto start_time = std::chrono::high_resolution_clock::now();
    update_cluster_nn(clusters);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

    UPDATE_NN_DURATIONS.push_back(duration.count());

    remove_secondary_clusters(merges, clusters);

    return RAC_r(base_arr, clusters);
}

std::vector<int> RAC(Eigen::MatrixXd& base_arr) {
    // Eigen::setNbThreads(8);
    // Transpose bare array for column access
    Eigen::MatrixXd new_base_arr = base_arr.transpose().colwise().normalized();

    std::vector<Cluster*> clusters;
    for (int i = 0; i < new_base_arr.cols(); ++i) {
        Cluster* cluster = new Cluster(i);
        clusters.push_back(cluster);
    }

    //set eigen threads
    Eigen::setNbThreads(8);
    calculate_initial_disimilarities(new_base_arr, clusters, MIN_DISTANCE);

    Eigen::setNbThreads(1);
    RAC_r(new_base_arr, clusters);

    Eigen::setNbThreads(8);
    std::vector<std::pair<int, int> > cluster_idx;
    for (Cluster* cluster : clusters) {
        if (cluster == nullptr) {
            continue;
        }

        for (int index : cluster->indices)  {
            cluster_idx.push_back(std::make_pair(index, cluster->id));
        }
    }

    std::sort(cluster_idx.begin(), cluster_idx.end());

    std::vector<int> cluster_labels;
    for (const auto& [index, cluster_id] : cluster_idx) {
        cluster_labels.push_back(cluster_id);
    }

    return cluster_labels;
}

