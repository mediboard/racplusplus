#include "pybind11/include/pybind11/pybind11.h"
#include "pybind11/include/pybind11/eigen.h"
#include "racplusplus.h"
#include <array>
#include <tuple>
#include <unordered_map>
#include <set>
#include <chrono>
#include <vector>
#include <map>
#include <iostream>
#include <thread>
#include <algorithm>
// #define EIGEN_DONT_PARALLELIZE
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <random>

//get number of processors
size_t getProcessorCount() {
    const auto NO_PROCESSORS = std::thread::hardware_concurrency();
    return NO_PROCESSORS != 0 ? static_cast<size_t>(NO_PROCESSORS) : static_cast<size_t>(8);
}

//--------pybind--------------


//----main
int main() {
    std::cout << "Starting Randomized RAC Test" << std::endl;
    std::cout << "Number of Processors Found for Program Use: " << getProcessorCount() << std::endl;

    // 5000 - 1061
    const int NO_POINTS = 10000;
    Eigen::MatrixXd test = generateRandomMatrix(NO_POINTS, 768, 10);

    // Shift and scale the values to the range 0-1
    test = (test + Eigen::MatrixXd::Constant(NO_POINTS, 768, 1.)) / 2.;

    // std::cout << test << std::endl;

    // Eigen::SparseMatrix<bool> connectivity(NO_POINTS, NO_POINTS);
    // for (size_t i=0; i<NO_POINTS; i++) {
    //     for (size_t j=0; j<NO_POINTS; j++) {
    //         connectivity.insert(i, j) = true;
    //     }
    // }
    Eigen::SparseMatrix<bool> connectivity;

    std::cout << "Actually running RAC now." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    //set up test
    double min_distance = .035;
    int batch_size = 1000;
    int no_processors = 0;
    int no_points = 10000;
    //actually run test
    std::vector<int> labels = RAC(test, min_distance, batch_size, connectivity, no_processors, no_points);
    
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "RAC Finished!" << std::endl;

    // Compute the duration
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    // Output duration
    std::cout << duration.count() << std::endl;

    // Output neighbor update durations
    std::cout << std::accumulate(UPDATE_NEIGHBOR_DURATIONS.begin(), UPDATE_NEIGHBOR_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // Output NN update durations
    std::cout << std::accumulate(UPDATE_NN_DURATIONS.begin(), UPDATE_NN_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // Output indices durations
    std::cout << std::accumulate(INDICES_DURATIONS.begin(), INDICES_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // Output merge durations
    std::cout << std::accumulate(MERGE_DURATIONS.begin(), MERGE_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // Output misc merge durations
    std::cout << std::accumulate(MISC_MERGE_DURATIONS.begin(), MISC_MERGE_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // Output number of clusters
    std::set<int> unique_labels(labels.begin(), labels.end());
    std::cout << unique_labels.size() << std::endl;

    // Output number of cosine calls
    // std::cout << NO_COSINE_CALLS << std::endl;

    // Output max cosine duration
    // std::cout << std::max_element(COSINE_DURATIONS.begin(), COSINE_DURATIONS.end())[0] << std::endl;

    // // Output total cosine duration
    // std::cout << std::accumulate(COSINE_DURATIONS.begin(), COSINE_DURATIONS.end(), 0.0) / 1000 << std::endl;

    // // Output average cosine duration
    // std::cout << std::accumulate(COSINE_DURATIONS.begin(), COSINE_DURATIONS.end(), 0.0) / COSINE_DURATIONS.size() << std::endl;
    return 0;
}

void stupid_pybind_test() {
    std::cout<<"this is a pybind test." << std::endl;

}

PYBIND11_MODULE(racplusplus, m){
    m.doc() = R"doc( 
        RACplusplus is a C++ optimized python package for performing
        reciprocal agglomerative clustering.

        Authors: Porter Hunley, Daniel Frees
        2023
    )doc";

    m.def("rac", &RAC, R"fdoc(
        Run RAC algorithm on a provided array of points.

        Returns an array of the group # each point was assigned to.
    )fdoc");

    m.def("test_rac", &main, R"fdoc(
        Testing function to run and time RAC's run in C++.
    )fdoc");

    m.def("stupid_pybind_test", &stupid_pybind_test, R"fdoc(
        Simple test function to see if pybind works.
    )fdoc");

    m.attr("__version__") = "0.9";
}


//-----------------------------

//---------------------Classes------------------------------------


Cluster::Cluster(int id)
    : id(id), will_merge(false) {
        std::vector<int> indices;
        indices.push_back(id);
        this->indices = indices;

        this->nn = -1;
    }


void Cluster::update_nn(float min_distance) {
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

    if (min <= min_distance) {
        this->nn = nn;
    } else {
        this->nn = -1;
    }
}

void Cluster::update_nn(Eigen::MatrixXd& distance_arr) {
    if (neighbors.size() == 0) {
        nn = -1;
        return;
    }

    double min = 1;
    int nn = -1;
    for (int neighbor : this->neighbors) {
        float dissimilarity = distance_arr(this->id, neighbor);
        if (dissimilarity < min) {
            min = dissimilarity;
            nn = neighbor;
        }
    }

    this->nn = nn;
}

//---------------------End Classes------------------------------------


//--------------------Helpers------------------------------------
// Function to generate a matrix filled with random numbers.
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

double get_arr_value(Eigen::MatrixXd& arr, int i, int j) {
    if (i > j) {
        return arr(j, i);
    }
    return arr(i, j);
}

void set_arr_value(Eigen::MatrixXd& arr, int i, int j, double value) {
    if (i > j) {
        arr(j, i) = value;
        return;
    }
    arr(i, j) = value;
}

void remove_secondary_clusters(std::vector<std::pair<int, int> >& merges, std::vector<Cluster*>& clusters) {
    for (const auto& merge : merges) {
        int secondary_id = merge.second;
        clusters[secondary_id] = nullptr;
    }
}
//--------------------End Helpers------------------------------------

//-----------------------Distance Calculations-------------------------
//Calculate pairwise cosines between two matrices
Eigen::MatrixXd pairwise_cosine(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b) {
    return Eigen::MatrixXd::Ones(array_a.cols(), array_b.cols()) - (array_a.transpose() * array_b);
}

//Averaged dissimilarity across two matrices (wrapper for pairwise distance calc + avging)
float calculate_weighted_dissimilarity(Eigen::MatrixXd points_a, Eigen::MatrixXd points_b) {
    Eigen::MatrixXd dissimilarity_matrix = pairwise_cosine(points_a, points_b);

    return static_cast<float>(dissimilarity_matrix.mean());
}

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges, 
    std::vector<Cluster*>& clusters,
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>>& sort_neighbor,
    std::vector<int>& update_neighbors,
    std::vector<std::vector<int>>& merging_arrays,
    const int NO_PROCESSORS) {
    auto start = std::chrono::high_resolution_clock::now();
    if (merges.size() / NO_PROCESSORS > 10) {
        parallel_merge_clusters(merges, clusters, NO_PROCESSORS, merging_arrays);
    } else {
        for (std::pair<int, int> merge : merges) {
            merge_cluster_apx(merge, clusters, merging_arrays[0]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    MERGE_DURATIONS.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    for (const auto& merge : merges) {
        int main = merge.first;
        int secondary = merge.second;

        clusters[main]->indices.insert(clusters[main]->indices.end(), clusters[secondary]->indices.begin(), clusters[secondary]->indices.end());

        for (size_t i=0; i < clusters[main]->neighbors_needing_updates.size(); i++) {
            int neighbor_idx = std::get<1>(clusters[main]->neighbors_needing_updates[i]);
            double dissimilarity = std::get<2>(clusters[main]->neighbors_needing_updates[i]);

            sort_neighbor[neighbor_idx].first = neighbor_idx;
            sort_neighbor[neighbor_idx].second.push_back(std::make_pair(main, dissimilarity));
        }
    }

    for (size_t i=0; i<sort_neighbor.size(); i++) {
        if (sort_neighbor[i].second.size() > 0) {
            update_cluster_neighbors(sort_neighbor[i], clusters, update_neighbors);
            sort_neighbor[i].second.clear();
        }
    }
}

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster*>& clusters,
    Eigen::MatrixXd& distance_arr,
    float min_distance,
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>>& sort_neighbor,
    std::vector<int>& update_neighbors,
    std::vector<std::vector<int>>& merging_arrays,
    const int NO_PROCESSORS) {

    if (merges.size() / NO_PROCESSORS > 10) {
        parallel_merge_clusters(merges, distance_arr, clusters, NO_PROCESSORS, min_distance, merging_arrays);
    } else {
        for (std::pair<int, int> merge : merges) {
            merge_cluster_apx(merge, clusters, merging_arrays[0], distance_arr, min_distance);
        }
    }

    for (const auto& merge : merges) {
        int main = merge.first;
        int secondary = merge.second;

        clusters[main]->indices.insert(clusters[main]->indices.end(), clusters[secondary]->indices.begin(), clusters[secondary]->indices.end());

        for (size_t i=0; i < clusters[main]->neighbors_needing_updates.size(); i++) {
            int neighbor_idx = std::get<1>(clusters[main]->neighbors_needing_updates[i]);
            double dissimilarity = std::get<2>(clusters[main]->neighbors_needing_updates[i]);

            sort_neighbor[neighbor_idx].first = neighbor_idx;
            sort_neighbor[neighbor_idx].second.push_back(std::make_pair(main, dissimilarity));
        }
    }

    for (size_t i=0; i<sort_neighbor.size(); i++) {
        if (sort_neighbor[i].second.size() > 0) {
            update_cluster_neighbors(sort_neighbor[i], clusters, distance_arr, min_distance, update_neighbors);
            sort_neighbor[i].second.clear();
        }
    }
}

Eigen::MatrixXd calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster*>& clusters,
    float min_distance) {
    Eigen::MatrixXd distance_mat = pairwise_cosine(base_arr, base_arr).array();

    for (size_t i=0; i<clusters.size(); i++) {
        double min = 2;
        int nn = -1;

        for (size_t j=0; j<clusters.size(); j++) {
            if (i == j) {
                distance_mat(i, j) = 2;
                continue;
            }

            double distance = distance_mat(i, j);
            if (distance <= min_distance) {
                clusters[i]->neighbors.push_back(j);

                if (distance < min) {
                    min = distance;
                    nn = j;
                }
            }
        }

        clusters[i] -> nn = nn;
    }

    return distance_mat;
}

void calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster*>& clusters,
    int batchSize,
    float min_distance,
    Eigen::SparseMatrix<bool>& connectivity) {

    for (int batchStart = 0; batchStart < clusters.size(); batchStart += batchSize) {
        int batchEnd = std::min(batchStart + batchSize, static_cast<int>(clusters.size()));
        Eigen::MatrixXd batch = base_arr.block(0, clusters[batchStart]->indices[0], base_arr.rows(), clusters[batchEnd - 1]->indices[0] - clusters[batchStart]->indices[0] + 1);

        Eigen::MatrixXd distance_mat = pairwise_cosine(base_arr, batch).array();
        for (int i = batchStart; i < batchEnd; ++i) {
            Cluster* cluster = clusters[i];
            Eigen::VectorXd distance_vec = distance_mat.col(i - batchStart);

            std::vector<int> neighbors;
            std::unordered_map<int, float> dissimilarities;
            for (int j = 0; j < static_cast<int>(distance_vec.size()); ++j) {
                // if (j != cluster->id && connectivity.coeff(cluster->id, j)) {
                if (j != cluster->id) {
                    dissimilarities[j] = distance_vec[j];
                    neighbors.push_back(j);
                }
            }

            cluster->neighbors = neighbors;
            cluster->dissimilarities = dissimilarities;

            distance_vec[cluster->id] = std::numeric_limits<float>::max(); // Masking
            auto min_position = std::min_element(distance_vec.data(), distance_vec.data() + distance_vec.size());

            int nearest_neighbor = std::distance(distance_vec.data(), min_position);
            cluster->nn = nearest_neighbor;
        }
    }
}

//-----------------------End Distance Calculations-------------------------

//-----------------------Merging Functions-----------------------------------
void merge_cluster_apx(
    std::pair<int, int>& merge,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& distance_arr,
    float min_distance) {

    Cluster* main_cluster = clusters[merge.first];
    Cluster* secondary_cluster = clusters[merge.second];

    std::vector<int> new_neighbors;

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
                if (merging_array[smallest_id] == 0) {
                    merging_array[smallest_id] = 1;
                    merging_neighbors.push_back(smallest_id);
                }
            } else {
                if (merging_array[id] == 0) {
                    static_neighbors.push_back(id);
                }

                ++merging_array[id];
            }
        }
    }

    std::vector<std::tuple<int, int, double> > needs_update;
    for (auto& static_id : static_neighbors) {
        double primary_dist = distance_arr(main_cluster->id, static_id);
        double secondary_dist = distance_arr(secondary_cluster->id, static_id);

        double avg_dist = (main_cluster->indices.size() * primary_dist + secondary_cluster->indices.size() * secondary_dist) / (main_cluster->indices.size() + secondary_cluster->indices.size());

        distance_arr(main_cluster->id, static_id) = avg_dist;
        if (avg_dist <= min_distance) {
            new_neighbors.push_back(static_id);
        }

        needs_update.push_back(std::make_tuple(main_cluster->id, static_id, avg_dist));

        merging_array[static_id] = 0;
    }

    for (auto& merging_id : merging_neighbors) {
        double main_primary_dist = distance_arr(main_cluster->id, merging_id); 
        double main_secondary_dist = distance_arr(secondary_cluster->id, merging_id);
        double main_avg_dist = (main_cluster->indices.size() * main_primary_dist + secondary_cluster->indices.size() * main_secondary_dist) / (main_cluster->indices.size() + secondary_cluster->indices.size());

        int secondary_merging_id = clusters[merging_id]->nn;
        double secondary_primary_dist = distance_arr(main_cluster->id, secondary_merging_id);
        double secondary_secondary_dist = distance_arr(secondary_cluster->id, secondary_merging_id);
        double secondary_avg_dist = (main_cluster->indices.size() * secondary_primary_dist + secondary_cluster->indices.size() * secondary_secondary_dist) / (main_cluster->indices.size() + secondary_cluster->indices.size());

        double avg_dist = (clusters[merging_id]->indices.size() * main_avg_dist + clusters[secondary_merging_id]->indices.size() * secondary_avg_dist) / (clusters[merging_id]->indices.size() + clusters[secondary_merging_id]->indices.size());

        if (avg_dist <= min_distance) {
            distance_arr(main_cluster->id, merging_id) = avg_dist;
            new_neighbors.push_back(merging_id);
        }

        merging_array[merging_id] = 0;
    }

    main_cluster->neighbors = new_neighbors;
    main_cluster->neighbors_needing_updates = needs_update;
}

void merge_cluster_apx(
    std::pair<int, int>& merge,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array) {

    Cluster* main_cluster = clusters[merge.first];
    Cluster* secondary_cluster = clusters[merge.second];

    std::vector<int> new_neighbors;

    std::unordered_map<int, float> new_dissimilarities;
    new_dissimilarities.reserve(main_cluster->dissimilarities.size() + secondary_cluster->dissimilarities.size());

    std::vector<int> static_neighbors;
    static_neighbors.reserve(main_cluster->neighbors.size() + secondary_cluster->neighbors.size());

    std::vector<int> merging_neighbors;
    merging_neighbors.reserve(main_cluster->neighbors.size() + secondary_cluster->neighbors.size());

    for (auto& id : main_cluster->neighbors) {
        if (id != main_cluster->id && id != secondary_cluster->id) {
            int smallest_id = id < clusters[id]->nn ? id : clusters[id]->nn;
            if (clusters[id]->will_merge) {
                if (merging_array[smallest_id] == 0) {
                    merging_neighbors.push_back(smallest_id);
                }

                merging_array[smallest_id]++;
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
                if (merging_array[smallest_id] == 0) {
                    merging_neighbors.push_back(smallest_id);
                }
                merging_array[smallest_id]++;

            } else {
                if (merging_array[id] == 0) {
                    static_neighbors.push_back(id);
                }
                ++merging_array[id];
            }
        }
    }

    std::vector<std::tuple<int, int, double> > needs_update;
    for (auto& static_id : static_neighbors) {
        float avg_dist = -1.0;

        if (merging_array[static_id] > 1) {
            float primary_dist = main_cluster->dissimilarities[static_id]; 
            float secondary_dist = secondary_cluster->dissimilarities[static_id];

            avg_dist = (main_cluster->indices.size() * primary_dist + secondary_cluster->indices.size() * secondary_dist) / (main_cluster->indices.size() + secondary_cluster->indices.size());

            new_dissimilarities[static_id] = avg_dist;
            new_neighbors.push_back(static_id);
        }

        needs_update.push_back(std::make_tuple(main_cluster->id, static_id, avg_dist));
        merging_array[static_id] = 0;
    }

    for (auto& merging_id : merging_neighbors) {
        float avg_dist = -1.0;

        if (merging_array[merging_id] > 3) {
            // Merging neighbors have the distance here
            float main_primary_dist = main_cluster->dissimilarities[merging_id]; 
            float main_secondary_dist = secondary_cluster->dissimilarities[merging_id];
            float main_avg_dist = (main_cluster->indices.size() * main_primary_dist + secondary_cluster->indices.size() * main_secondary_dist) / (main_cluster->indices.size() + secondary_cluster->indices.size());

            int secondary_merging_id = clusters[merging_id]->nn;
            float secondary_primary_dist = main_cluster->dissimilarities[secondary_merging_id];
            float secondary_secondary_dist = secondary_cluster->dissimilarities[secondary_merging_id];
            float secondary_avg_dist = (main_cluster->indices.size() * secondary_primary_dist + secondary_cluster->indices.size() * secondary_secondary_dist) / (main_cluster->indices.size() + secondary_cluster->indices.size());

            float avg_dist = (clusters[merging_id]->indices.size() * main_avg_dist + clusters[secondary_merging_id]->indices.size() * secondary_avg_dist) / (clusters[merging_id]->indices.size() + clusters[secondary_merging_id]->indices.size());

            new_neighbors.push_back(merging_id);
            new_dissimilarities[merging_id] = avg_dist;

        } 

        merging_array[merging_id] = 0;
    }

    main_cluster->neighbors = new_neighbors;
    main_cluster->dissimilarities = new_dissimilarities;
    main_cluster->neighbors_needing_updates = needs_update;
}

void merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array) {
    for (std::pair<int, int> merge : merges) {
        merge_cluster_apx(merge, clusters, merging_array);
    }
}

void merge_clusters_dist(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& distance_arr,
    float min_distance) {
    for (std::pair<int, int> merge : merges) {
        merge_cluster_apx(merge, clusters, merging_array, distance_arr, min_distance);
    }
}

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges, 
    std::vector<Cluster*>& clusters,
    size_t no_threads,
    std::vector<std::vector<int>>& merging_arrays) {

    std::vector<std::thread> threads;

    std::vector<std::vector<std::pair<int, int>>> merge_chunks(no_threads);

    size_t chunk_size = merges.size() / no_threads;
    size_t remainder = merges.size() % no_threads; 

    size_t start = 0, end = 0;
    for (size_t i = 0; i < no_threads; i++) {
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

    for (size_t i=0; i<no_threads; i++) {
        std::thread merge_thread = std::thread(
            merge_clusters,
            std::ref(merge_chunks[i]),
            std::ref(clusters),
            std::ref(merging_arrays[i]));

        threads.push_back(std::move(merge_thread));
    }

    for (size_t i=0; i<no_threads; i++) {
        threads[i].join();
    }
}

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    Eigen::MatrixXd& distance_arr,
    std::vector<Cluster*>& clusters,
    size_t no_threads,
    float min_distance,
    std::vector<std::vector<int>>& merging_arrays) {

    std::vector<std::thread> threads;

    std::vector<std::vector<std::pair<int, int>>> merge_chunks(no_threads);

    size_t chunk_size = merges.size() / no_threads;
    size_t remainder = merges.size() % no_threads; 

    size_t start = 0, end = 0;
    for (size_t i = 0; i < no_threads; i++) {
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

    for (size_t i=0; i<no_threads; i++) {
        std::thread merge_thread = std::thread(
            merge_clusters_dist,
            std::ref(merge_chunks[i]),
            std::ref(clusters),
            std::ref(merging_arrays[i]),
            std::ref(distance_arr),
            min_distance);

        threads.push_back(std::move(merge_thread));
    }

    for (size_t i=0; i<no_threads; i++) {
        threads[i].join();
    }
}
//-----------------------End Merging Functions-----------------------------------

//-----------------------Updating Nearest Neighbors-----------------------------------

void update_cluster_neighbors(
    std::pair<int, std::vector<std::pair<int, double> > >& update_chunk,
    std::vector<Cluster*>& clusters,
    std::vector<int>& update_neighbors) {
    Cluster* other_cluster = clusters[update_chunk.first];
    // start timer
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int> new_neighbors;
    std::vector<int> all_looped_neighbors;
    for (size_t i=0; i<update_chunk.second.size(); i++) {
        int neighbor_id = update_chunk.second[i].first;
        int neighbor_nn_id = clusters[neighbor_id]->nn;
        float dissimilarity = update_chunk.second[i].second;

        update_neighbors[neighbor_id] = 1;
        update_neighbors[neighbor_nn_id] = -1;

        other_cluster->dissimilarities[neighbor_id] = dissimilarity;
        if (dissimilarity >= 0) {
            new_neighbors.push_back(neighbor_id);
        }

        all_looped_neighbors.push_back(neighbor_id);
        all_looped_neighbors.push_back(neighbor_nn_id);
    }

    for (size_t i=0; i<other_cluster->neighbors.size(); i++) {
        int neighbor_id = other_cluster->neighbors[i];
        if (update_neighbors[neighbor_id] == 0) {
            new_neighbors.push_back(neighbor_id);
        }
        all_looped_neighbors.push_back(neighbor_id);
    }

    for (size_t i=0; i<all_looped_neighbors.size(); i++) {
        update_neighbors[all_looped_neighbors[i]] = 0;
    }

    other_cluster->neighbors = new_neighbors;

    // end timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    UPDATE_NEIGHBOR_DURATIONS.push_back(duration);
}

void update_cluster_neighbors(
    std::pair<int, std::vector<std::pair<int, double> > >& update_chunk,
    std::vector<Cluster*>& clusters,
    Eigen::MatrixXd& distance_arr,
    float min_distance,
    std::vector<int>& update_neighbors) {
    Cluster* other_cluster = clusters[update_chunk.first];

    std::vector<int> new_neighbors;
    std::vector<int> all_looped_neighbors;
    for (size_t i=0; i<update_chunk.second.size(); i++) {
        int neighbor_id = update_chunk.second[i].first;
        int neighbor_nn_id = clusters[neighbor_id]->nn;
        double dissimilarity = update_chunk.second[i].second;
        

        update_neighbors[neighbor_id] = 1;
        update_neighbors[neighbor_nn_id] = -1;

        distance_arr(other_cluster->id, neighbor_id) = dissimilarity;
        if (dissimilarity <= min_distance) {
            new_neighbors.push_back(neighbor_id);
        } 

        all_looped_neighbors.push_back(neighbor_id);
        all_looped_neighbors.push_back(neighbor_nn_id);
    }

    for (size_t i=0; i<other_cluster->neighbors.size(); i++) {
        int neighbor_id = other_cluster->neighbors[i];
        if (update_neighbors[neighbor_id] == 0) {
            new_neighbors.push_back(neighbor_id);
        }
        all_looped_neighbors.push_back(neighbor_id);
    }

    // Clear array and find NN
    for (size_t i=0; i<all_looped_neighbors.size(); i++) {
        update_neighbors[all_looped_neighbors[i]] = 0;
    }

    other_cluster->neighbors = new_neighbors;
}

void update_cluster_neighbors_p(
    std::vector<std::pair<int, std::vector<std::pair<int, double> > > >& updates,
    std::vector<Cluster*>& clusters, 
    std::vector<int>& update_neighbors) {
    for (auto& update: updates) {
        update_cluster_neighbors(update, clusters, update_neighbors);
    }
}   

void parallel_update_clusters(
    std::vector<std::pair<int, std::vector<std::pair<int, double> > > >& updates,
    std::vector<Cluster*>& clusters,
    size_t no_threads, 
    std::vector<int>& update_neighbors) {

    std::vector<std::thread> threads;
    std::vector<std::vector<std::pair<int, std::vector<std::pair<int, double> > > > > update_chunks(no_threads);

    size_t chunk_size = updates.size() / no_threads;
    size_t remainder = updates.size() % no_threads; 

    size_t start = 0, end = 0;
    for (size_t i = 0; i < no_threads; i++) {
        end = start + chunk_size;
        if (i < remainder) { // distribute the remainder among the first "remainder" chunks
            end++;
        }

        if (end <= updates.size()) {
            update_chunks[i] = std::vector<std::pair<int, std::vector<std::pair<int, double> > > >(updates.begin() + start, updates.begin() + end);
        }
        start = end;
    }

    for (size_t i=0; i<no_threads; i++) {
        std::thread update_thread = std::thread(
            update_cluster_neighbors_p,
            std::ref(update_chunks[i]),
            std::ref(clusters),
            std::ref(update_neighbors));

        threads.push_back(std::move(update_thread));
    }

    for (size_t i=0; i<no_threads; i++) {
        threads[i].join();
    }
}

void update_cluster_nn(
    std::vector<Cluster*>& clusters,
    Eigen::MatrixXd& distance_arr,
    float min_disitance) {
    for (Cluster* cluster : clusters) {
        if (cluster == nullptr) {
            continue;
        }

        if (cluster->will_merge || (cluster->nn != -1 and clusters[cluster->nn] != nullptr and clusters[cluster->nn]->will_merge)) {
            if (distance_arr.size() == 0) {
                cluster->update_nn(min_disitance);
            } else {
                cluster->update_nn(distance_arr);
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

        cluster -> will_merge = false;

        if (cluster->nn != -1 && clusters[cluster->nn] != nullptr) {
            cluster->will_merge = (clusters[cluster->nn]->nn == cluster->id);
        }

        if (cluster -> will_merge && cluster->id < cluster->nn) {
            reciprocal_nn.push_back(std::make_pair(cluster->id, cluster->nn));
        }
    }

    return reciprocal_nn;
}
//-----------------------End Updating Nearest Neighbors-----------------------------------

//--------------------------------------RAC Functions--------------------------------------
void RAC_i(
    std::vector<Cluster*>& clusters, 
    float min_distance, 
    Eigen::MatrixXd& distance_arr, 
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>>& sort_neighbor,
    std::vector<int>& update_neighbors,
    std::vector<std::vector<int>>& merging_arrays,
    const int NO_PROCESSORS) {
    while (true) {
        std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters);
        if (merges.size() == 0) {
            break;
        }

        if (distance_arr.size() != 0) {
            update_cluster_dissimilarities(merges, clusters, distance_arr, min_distance, sort_neighbor, update_neighbors, merging_arrays, NO_PROCESSORS);
        } else {
            update_cluster_dissimilarities(merges, clusters, sort_neighbor, update_neighbors, merging_arrays, NO_PROCESSORS);
        }

        update_cluster_nn(clusters, distance_arr, min_distance);

        remove_secondary_clusters(merges, clusters);
    }
}

std::vector<int> RAC(
    Eigen::MatrixXd& base_arr,
    float min_distance,
    int batchSize,
    Eigen::SparseMatrix<bool>& connectivity,
    int no_processors = 0,
    const int NO_POINTS = 10000) {

    const int NO_PROCESSORS = (no_processors != 0) ? no_processors : getProcessorCount();

    // Dynamically allocate sort_neighbor
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>> sort_neighbor(NO_POINTS);
    // Dynamically allocate update_neighbors
    std::vector<int> update_neighbors(NO_POINTS);
    // Dynamically allocate merging_arrays
    std::vector<std::vector<int>> merging_arrays(NO_PROCESSORS, std::vector<int>(NO_POINTS));

    base_arr = base_arr.transpose().colwise().normalized().eval();

    // start timer
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Cluster*> clusters;
    for (long i = 0; i < base_arr.cols(); ++i) {
        Cluster* cluster = new Cluster(i);
        clusters.push_back(cluster);
    }
    Eigen::setNbThreads(8);
    Eigen::MatrixXd distance_arr;

    if (connectivity.size() == 0) {
        calculate_initial_dissimilarities(base_arr, clusters, batchSize, min_distance, connectivity);
    } else {
        distance_arr = calculate_initial_dissimilarities(base_arr, clusters, min_distance);
    }

    // end timer
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "Time taken to calculate initial dissimilarities: " << duration << "ms" << std::endl;

    RAC_i(clusters, min_distance, distance_arr, sort_neighbor, update_neighbors, merging_arrays, NO_PROCESSORS);

    // Set Eigen Threads according to Number of processors
    Eigen::setNbThreads(NO_PROCESSORS);
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
//--------------------------------------End RAC Functions--------------------------------------
