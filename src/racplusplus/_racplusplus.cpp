#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
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
#include <numeric>

#include "_racplusplus.h"

//get number of processors
size_t getProcessorCount() {
    const auto NO_PROCESSORS = std::thread::hardware_concurrency();
    return NO_PROCESSORS != 0 ? static_cast<size_t>(NO_PROCESSORS) : static_cast<size_t>(8);
}

std::string vectorToString(const std::vector<std::pair<int, int>>& merges) {
    std::ostringstream oss;
    oss << "[";
    for (auto it = merges.begin(); it != merges.end(); ++it) {
        oss << "(" << it->first << ", " << it->second << ")";
        if (std::next(it) != merges.end()) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}

//----main
int main() {
    std::cout << std::endl;
    std::cout << "Starting Randomized RAC Test" << std::endl;
    std::cout << "Number of Processors Found for Program Use: " << getProcessorCount() << std::endl;
    // 5000 - 1061
    const int NO_POINTS = 20000;
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
    std::cout << "Actually running RAC now..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    //set up test
    double max_merge_distance = .035;
    int batch_size = 100;
    int no_processors = 0;
    //actually run test
    std::vector<int> labels = RAC(test, max_merge_distance, connectivity, "full", batch_size, no_processors);
    
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
    std::cout << std::endl;
    return 0;
}

//---------------------Classes------------------------------------


Cluster::Cluster(int id)
    : id(id), will_merge(false) {
        std::vector<int> indices;
        indices.push_back(id);
        this->indices = indices;

        this->nn = -1;
    }


void Cluster::update_nn(double max_merge_distance) {
    if (neighbors.size() == 0) {
        nn = -1;
        return;
    }

    double min = 1;
    int nn = -1;

    for (int neighbor : this->neighbors) {
        double dissimilarity = this->dissimilarities[neighbor];
        if (dissimilarity < min) {
            min = dissimilarity;
            nn = neighbor;
        }
    }

    if (min < max_merge_distance) {
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
        double dissimilarity = distance_arr(this->id, neighbor);
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

    int numRows = mat.rows();
    int numCols = mat.cols();
    for(int i=0; i<numRows; ++i) {
        for(int j=0; j<numCols; ++j) {
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
double calculate_weighted_dissimilarity(Eigen::MatrixXd points_a, Eigen::MatrixXd points_b) {
    Eigen::MatrixXd dissimilarity_matrix = pairwise_cosine(points_a, points_b);

    return static_cast<double>(dissimilarity_matrix.mean());
}

void consolidate_indices(
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>>& sort_neighbor_arr,
    std::vector<std::pair<int, int> >& merges, 
    std::vector<Cluster*> clusters) {
    
    for (const auto& merge : merges) {
        int main = merge.first;
        int secondary = merge.second;

        clusters[main]->indices.insert(clusters[main]->indices.end(), clusters[secondary]->indices.begin(), clusters[secondary]->indices.end());

        for (size_t i=0; i < clusters[main]->neighbors_needing_updates.size(); i++) {
            int neighbor_idx = std::get<1>(clusters[main]->neighbors_needing_updates[i]);
            double dissimilarity = std::get<2>(clusters[main]->neighbors_needing_updates[i]);

            sort_neighbor_arr[neighbor_idx].first = neighbor_idx;
            sort_neighbor_arr[neighbor_idx].second.push_back(std::make_pair(main, dissimilarity));
        }
    }
}

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges, 
    std::vector<Cluster*>& clusters,
    const int NO_PROCESSORS,
    Eigen::MatrixXd& base_arr) {

    static std::vector<std::vector<int>> merging_arrays(NO_PROCESSORS, std::vector<int>(clusters.size()));

    auto start = std::chrono::high_resolution_clock::now();
    if (merges.size() / NO_PROCESSORS > 10) {
        parallel_merge_clusters(merges, clusters, NO_PROCESSORS, merging_arrays, base_arr);
    } else {
        for (std::pair<int, int> merge : merges) {
            merge_cluster_compute_linkage(merge, clusters, merging_arrays[0], base_arr);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    MERGE_DURATIONS.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    static std::vector<std::pair<int, std::vector<std::pair<int, double>>>> sort_neighbor_arr(clusters.size());
    consolidate_indices(sort_neighbor_arr, merges, clusters);

    static std::vector<int> update_neighbors_arr(clusters.size());
    for (size_t i=0; i<sort_neighbor_arr.size(); i++) {
        if (sort_neighbor_arr[i].second.size() > 0) {
            update_cluster_neighbors(sort_neighbor_arr[i], clusters, update_neighbors_arr);
            sort_neighbor_arr[i].second.clear();
        }
    }
}

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges, 
    std::vector<Cluster*>& clusters,
    const int NO_PROCESSORS) {

    static std::vector<std::vector<int>> merging_arrays(NO_PROCESSORS, std::vector<int>(clusters.size()));

    auto start = std::chrono::high_resolution_clock::now();
    if (merges.size() / NO_PROCESSORS > 10) {
        parallel_merge_clusters(merges, clusters, NO_PROCESSORS, merging_arrays);
    } else {
        for (std::pair<int, int> merge : merges) {
            merge_cluster_symmetric_linkage(merge, clusters, merging_arrays[0]);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    MERGE_DURATIONS.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    static std::vector<std::pair<int, std::vector<std::pair<int, double>>>> sort_neighbor_arr(clusters.size());
    consolidate_indices(sort_neighbor_arr, merges, clusters);

    static std::vector<int> update_neighbors_arr(clusters.size());
    for (size_t i=0; i<sort_neighbor_arr.size(); i++) {
        if (sort_neighbor_arr[i].second.size() > 0) {
            update_cluster_neighbors(sort_neighbor_arr[i], clusters, update_neighbors_arr);
            sort_neighbor_arr[i].second.clear();
        }
    }
}

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster*>& clusters,
    Eigen::MatrixXd& distance_arr,
    double max_merge_distance,
    const int NO_PROCESSORS) {

    static std::vector<std::vector<int>> merging_arrays(NO_PROCESSORS, std::vector<int>(clusters.size()));

    if (merges.size() / NO_PROCESSORS > 10) {
        parallel_merge_clusters(merges, distance_arr, clusters, NO_PROCESSORS, max_merge_distance, merging_arrays);
    } else {
        for (std::pair<int, int> merge : merges) {
            merge_cluster_full(merge, clusters, merging_arrays[0], distance_arr, max_merge_distance);
        }
    }

    static std::vector<std::pair<int, std::vector<std::pair<int, double>>>> sort_neighbor_arr(clusters.size());
    consolidate_indices(sort_neighbor_arr, merges, clusters);

    static std::vector<int> update_neighbors_arr(clusters.size());
    for (size_t i=0; i<sort_neighbor_arr.size(); i++) {
        if (sort_neighbor_arr[i].second.size() > 0) {
            update_cluster_neighbors(sort_neighbor_arr[i], clusters, distance_arr, max_merge_distance, update_neighbors_arr);
            sort_neighbor_arr[i].second.clear();
        }
    }
}

Eigen::MatrixXd calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster*>& clusters,
    double max_merge_distance) {
    Eigen::MatrixXd distance_mat = pairwise_cosine(base_arr, base_arr).array();

    size_t clusterSize = clusters.size();
    for (size_t i=0; i<clusterSize; i++) {
        double min = 2;
        int nn = -1;

        auto currentCluster = clusters[i];

        for (size_t j=0; j<clusterSize; j++) {
            if (i == j) {
                distance_mat(i, j) = 2;
                continue;
            }

            double distance = distance_mat(i, j);
            if (distance < max_merge_distance) {
                currentCluster->neighbors.push_back(j);

                if (distance < min) {
                    min = distance;
                    nn = j;
                }
            }
        }

        currentCluster -> nn = nn;
    }

    return distance_mat;
}

void calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster*>& clusters,
    Eigen::SparseMatrix<bool>& connectivity,
    double max_merge_distance,
    int batch_size) {

    int clustersSize = static_cast<int>(clusters.size());
    for (int batchStart = 0; batchStart < clustersSize; batchStart += batch_size) {
        int batchEnd = std::min(batchStart + batch_size, clustersSize);
        Eigen::MatrixXd batch = base_arr.block(0, clusters[batchStart]->indices[0], base_arr.rows(), clusters[batchEnd - 1]->indices[0] - clusters[batchStart]->indices[0] + 1);

        Eigen::MatrixXd distance_mat = pairwise_cosine(base_arr, batch).array();
        for (int i = batchStart; i < batchEnd; ++i) {
            Cluster* cluster = clusters[i];
            Eigen::VectorXd distance_vec = distance_mat.col(i - batchStart);

            std::vector<int> neighbors;
            std::unordered_map<int, double> dissimilarities;

            int distanceVecSize = static_cast<int>(distance_vec.size());
            int nearest_neighbor = -1;
            double min = 2;
            for (int j = 0; j < distanceVecSize; ++j) {
                if (j != cluster->id && connectivity.coeff(cluster->id, j)) {
                    dissimilarities[j] = distance_vec[j];
                    neighbors.push_back(j);

                    if (distance_vec[j] < min && distance_vec[j] < max_merge_distance) {
                        min = distance_vec[j];
                        nearest_neighbor = j;
                    }
                }
            }

            cluster->neighbors = neighbors;
            cluster->dissimilarities = dissimilarities;

            distance_vec[cluster->id] = std::numeric_limits<double>::max(); // Masking
            cluster->nn = nearest_neighbor;
        }
    }
}

//-----------------------End Distance Calculations-------------------------

//-----------------------Merging Functions-----------------------------------
// Function that fills in distance holes
double get_cluster_distances(
    Cluster* main_cluster,
    std::vector<int>& other_cluster_idxs,
    int other_cluster_id,
    Eigen::MatrixXd& base_arr) {

    std::vector<int> recalculate_idxs;
    double main_dist;
    int no_calc_dists;

    if (main_cluster->dissimilarities.find(other_cluster_id) != main_cluster->dissimilarities.end()) {
        return main_cluster->dissimilarities[other_cluster_id];

    } else {
        Eigen::MatrixXd full_main = base_arr(Eigen::all, main_cluster->indices);
        Eigen::MatrixXd full_other = base_arr(Eigen::all, other_cluster_idxs);
        double dist = pairwise_cosine(full_main, full_other).mean();

        return dist;
    }

    double rolling_dist = 0.0;
    int no_dists = 0.0;
    for (unsigned long i=0; i < other_cluster_idxs.size(); ++i) {
        int idx = other_cluster_idxs[i];

        if (main_cluster->dissimilarities.find(idx) == main_cluster->dissimilarities.end()) {
            recalculate_idxs.push_back(idx);
        } else {
            rolling_dist += main_cluster->dissimilarities[idx];
            ++no_dists;
        }
    }

    Eigen::MatrixXd full_main = base_arr(Eigen::all, main_cluster->indices);
    Eigen::MatrixXd partial_other = base_arr(Eigen::all, recalculate_idxs);

    Eigen::VectorXd new_dists = pairwise_cosine(full_main, partial_other).colwise().mean();
    double new_dists_sum = new_dists.sum();
    int new_dists_no = new_dists.size();

    return (rolling_dist + new_dists_sum) / (no_dists + new_dists_no);
}

std::pair<std::vector<int>, std::vector<int> > split_neighbors(
    Cluster* main_cluster,
    Cluster* secondary_cluster,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array) {
    
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

    return std::make_pair(static_neighbors, merging_neighbors);
}

// Merges with the full distance array
void merge_cluster_full(
    std::pair<int, int>& merge,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& distance_arr,
    double max_merge_distance) {

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
        if (avg_dist < max_merge_distance) {
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

        if (avg_dist < max_merge_distance) {
            distance_arr(main_cluster->id, merging_id) = avg_dist;
            new_neighbors.push_back(merging_id);
        }

        merging_array[merging_id] = 0;
    }

    main_cluster->neighbors = new_neighbors;
    main_cluster->neighbors_needing_updates = needs_update;
}

void merge_cluster_symmetric_linkage(
    std::pair<int, int>& merge,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array) {

    Cluster* main_cluster = clusters[merge.first];
    Cluster* secondary_cluster = clusters[merge.second];

    std::vector<int> new_neighbors;
    std::unordered_map<int, double> new_dissimilarities;
    std::vector<std::tuple<int, int, double> > needs_update;

    std::vector<int> static_neighbors;
    std::vector<int> merging_neighbors;
    std::tie(static_neighbors, merging_neighbors) = split_neighbors(main_cluster, secondary_cluster, clusters, merging_array);

    for (auto& id : static_neighbors) {
        double main_dist = 0.0;
        double secondary_dist = 0.0;

        // Using merging array to minimize map lookups
        if (merging_array[id] > 1) { // Both clusters share this neighbor
            main_dist = main_cluster->dissimilarities[id];
            secondary_dist = secondary_cluster->dissimilarities[id];
        } else { // only one cluster has this neighbor
            if (main_cluster->dissimilarities.find(id) != main_cluster->dissimilarities.end()) {
                main_dist = main_cluster->dissimilarities[id];
                secondary_dist = main_dist;
            } else {
                secondary_dist = secondary_cluster->dissimilarities[id];
                main_dist = secondary_dist;
            }
        }

        int no_main = main_cluster->indices.size();
        int no_secondary = secondary_cluster->indices.size();
        double new_dist = (main_dist * no_main + secondary_dist * no_secondary) / (no_main + no_secondary); 

        new_neighbors.push_back(id);
        new_dissimilarities[id] = new_dist;
        merging_array[id] = 0;

        needs_update.push_back(std::make_tuple(main_cluster->id, id, new_dist));
    }

    for (auto& id : merging_neighbors) {
        // First focus on other
        double main_other_dist = 0.0;
        double secondary_other_dist = 0.0;

        std::vector<double> weighted_dists;
        std::vector<int> weighted_no;

        Cluster* other_cluster = clusters[id];

        bool main_has_other = main_cluster->dissimilarities.find(id) != main_cluster->dissimilarities.end();
        bool secondary_has_other = secondary_cluster->dissimilarities.find(id) != secondary_cluster->dissimilarities.end();

        if (main_has_other) {
            main_other_dist = main_cluster->dissimilarities[id];
            weighted_dists.push_back(main_other_dist * (main_cluster->indices.size() + other_cluster->indices.size()));
            weighted_no.push_back(main_cluster->indices.size() + other_cluster->indices.size());
        }

        if (secondary_has_other) {
            secondary_other_dist = secondary_cluster->dissimilarities[id];
            weighted_dists.push_back(secondary_other_dist * (secondary_cluster->indices.size() + other_cluster->indices.size()));
            weighted_no.push_back(secondary_cluster->indices.size() + other_cluster->indices.size());
        }

        int nn = clusters[id]->nn;
        double main_nn_dist = 0.0;
        double secondary_nn_dist = 0.0;

        bool main_has_nn = main_cluster->dissimilarities.find(nn) != main_cluster->dissimilarities.end();
        bool secondary_has_nn = secondary_cluster->dissimilarities.find(nn) != secondary_cluster->dissimilarities.end();

        if (main_has_nn) {
            main_nn_dist = main_cluster->dissimilarities[nn];
            weighted_dists.push_back(main_nn_dist * (main_cluster->indices.size() + clusters[nn]->indices.size()));
            weighted_no.push_back(main_cluster->indices.size() + clusters[nn]->indices.size());
        }

        if (secondary_has_nn) {
            secondary_nn_dist = secondary_cluster->dissimilarities[nn];
            weighted_dists.push_back(secondary_nn_dist * (secondary_cluster->indices.size() + clusters[nn]->indices.size()));
            weighted_no.push_back(secondary_cluster->indices.size() + clusters[nn]->indices.size());
        }

        double merge_dist = std::accumulate(weighted_dists.begin(), weighted_dists.end(), 0.0) / std::accumulate(weighted_no.begin(), weighted_no.end(), 0.0);

        new_neighbors.push_back(id);
        new_dissimilarities[id] = merge_dist;
        merging_array[id] = 0; 
    }

    main_cluster->neighbors = new_neighbors;
    main_cluster->dissimilarities = new_dissimilarities;
    main_cluster->neighbors_needing_updates = needs_update;
}

// Computes missing edges on the fly for a more balanced tree
void merge_cluster_compute_linkage(
    std::pair<int, int>& merge,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& base_arr) {

    Cluster* main_cluster = clusters[merge.first];
    Cluster* secondary_cluster = clusters[merge.second];

    std::vector<int> new_neighbors;

    std::unordered_map<int, double> new_dissimilarities;
    new_dissimilarities.reserve(main_cluster->dissimilarities.size() + secondary_cluster->dissimilarities.size());

    std::vector<int> static_neighbors;
    std::vector<int> merging_neighbors;
    std::tie(static_neighbors, merging_neighbors) = split_neighbors(main_cluster, secondary_cluster, clusters, merging_array);

    // std::cout << "main cluster: " << main_cluster->id << std::endl;
    std::vector<std::tuple<int, int, double> > needs_update;
    for (auto& static_id : static_neighbors) {
        double avg_dist = -1.0;

        double main_dist = get_cluster_distances(main_cluster, clusters[static_id]->indices, static_id, base_arr);
        double secondary_dist = get_cluster_distances(secondary_cluster, clusters[static_id]->indices, static_id, base_arr);

        avg_dist = (main_cluster->indices.size() * main_dist + secondary_cluster->indices.size() * secondary_dist) / (main_cluster->indices.size() + secondary_cluster->indices.size());
        // std::cout << "other avg_dist: " << avg_dist << std::endl;

        needs_update.push_back(std::make_tuple(main_cluster->id, static_id, avg_dist));
        new_neighbors.push_back(static_id);
        new_dissimilarities[static_id] = avg_dist;
        merging_array[static_id] = 0;
    }

    for (auto& merging_id : merging_neighbors) {
        double avg_dist = -1.0;

        double main_primary_dist = get_cluster_distances(main_cluster, clusters[merging_id]->indices, merging_id, base_arr);
        double main_secondary_dist = get_cluster_distances(secondary_cluster, clusters[merging_id]->indices, merging_id, base_arr);
        double main_avg_dist = (main_cluster->indices.size() * main_primary_dist + secondary_cluster->indices.size() * main_secondary_dist) / (main_cluster->indices.size() + secondary_cluster->indices.size());

        int secondary_merging_id = clusters[merging_id]->nn;
        double secondary_primary_dist = get_cluster_distances(main_cluster, clusters[secondary_merging_id]->indices, secondary_merging_id, base_arr);
        double secondary_secondary_dist = get_cluster_distances(secondary_cluster, clusters[secondary_merging_id]->indices, secondary_merging_id, base_arr);
        double secondary_avg_dist = (main_cluster->indices.size() * secondary_primary_dist + secondary_cluster->indices.size() * secondary_secondary_dist) / (main_cluster->indices.size() + secondary_cluster->indices.size());

        avg_dist = (clusters[merging_id]->indices.size() * main_avg_dist + clusters[secondary_merging_id]->indices.size() * secondary_avg_dist) / (clusters[merging_id]->indices.size() + clusters[secondary_merging_id]->indices.size());
        // std::cout << "other avg_dist: " << avg_dist << std::endl;

        new_neighbors.push_back(merging_id);
        new_dissimilarities[merging_id] = avg_dist;

        merging_array[merging_id] = 0;
    }

    main_cluster->neighbors = new_neighbors;
    main_cluster->dissimilarities = new_dissimilarities;
    main_cluster->neighbors_needing_updates = needs_update;
}

void merge_clusters_symmetric(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array) {
    
    for (auto& merge : merges) {
        merge_cluster_symmetric_linkage(merge, clusters, merging_array);
    }
}

void merge_clusters_compute(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& base_arr) {
    for (std::pair<int, int> merge : merges) {
        merge_cluster_compute_linkage(merge, clusters, merging_array, base_arr);
    }
}

void merge_clusters_full(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& distance_arr,
    double max_merge_distance) {
    for (std::pair<int, int> merge : merges) {
        merge_cluster_full(merge, clusters, merging_array, distance_arr, max_merge_distance);
    }
}

std::vector<std::vector<std::pair<int, int> > > chunk_merges(std::vector<std::pair<int, int> >& merges, size_t no_threads) {
    std::vector<std::vector<std::pair<int, int> > > merge_chunks(no_threads);

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
            merge_chunks[i] = std::vector<std::pair<int, int> >(merges.begin() + start, merges.begin() + end);
        } 
        start = end;
    }

    return merge_chunks;
}

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges, 
    std::vector<Cluster*>& clusters,
    size_t no_threads,
    std::vector<std::vector<int>>& merging_arrays) {
    
    std::vector<std::thread> threads;

    std::vector<std::vector<std::pair<int, int>>> merge_chunks;
    merge_chunks = chunk_merges(merges, no_threads);

    for (size_t i=0; i<no_threads; i++) {
        std::thread merge_thread = std::thread(
            merge_clusters_symmetric,
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
    std::vector<Cluster*>& clusters,
    size_t no_threads,
    std::vector<std::vector<int>>& merging_arrays,
    Eigen::MatrixXd& base_arr) {

    std::vector<std::thread> threads;

    std::vector<std::vector<std::pair<int, int>>> merge_chunks;
    merge_chunks = chunk_merges(merges, no_threads);

    for (size_t i=0; i<no_threads; i++) {
        std::thread merge_thread = std::thread(
            merge_clusters_compute,
            std::ref(merge_chunks[i]),
            std::ref(clusters),
            std::ref(merging_arrays[i]),
            std::ref(base_arr));

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
    double max_merge_distance,
    std::vector<std::vector<int>>& merging_arrays) {

    std::vector<std::thread> threads;

    std::vector<std::vector<std::pair<int, int>>> merge_chunks;
    merge_chunks = chunk_merges(merges, no_threads);

    for (size_t i=0; i<no_threads; i++) {
        std::thread merge_thread = std::thread(
            merge_clusters_full,
            std::ref(merge_chunks[i]),
            std::ref(clusters),
            std::ref(merging_arrays[i]),
            std::ref(distance_arr),
            max_merge_distance);

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

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<int> new_neighbors;
    std::vector<int> all_looped_neighbors;
    for (size_t i=0; i<update_chunk.second.size(); i++) {
        int neighbor_id = update_chunk.second[i].first;
        int neighbor_nn_id = clusters[neighbor_id]->nn;
        double dissimilarity = update_chunk.second[i].second;

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

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    UPDATE_NEIGHBOR_DURATIONS.push_back(duration);
}

void update_cluster_neighbors(
    std::pair<int, std::vector<std::pair<int, double> > >& update_chunk,
    std::vector<Cluster*>& clusters,
    Eigen::MatrixXd& distance_arr,
    double max_merge_distance,
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
        if (dissimilarity < max_merge_distance) {
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
    double max_merge_distance) {
    for (Cluster* cluster : clusters) {
        if (cluster == nullptr) {
            continue;
        }

        if (cluster->will_merge || (cluster->nn != -1 and clusters[cluster->nn] != nullptr and clusters[cluster->nn]->will_merge)) {
            cluster->update_nn(max_merge_distance);
        }
    }
}

void update_cluster_nn(
    std::vector<Cluster*>& clusters,
    Eigen::MatrixXd& distance_arr,
    double max_merge_distance) {
    for (Cluster* cluster : clusters) {
        if (cluster == nullptr) {
            continue;
        }

        if (cluster->will_merge || (cluster->nn != -1 and clusters[cluster->nn] != nullptr and clusters[cluster->nn]->will_merge)) {
            cluster->update_nn(distance_arr);
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
    double max_merge_distance, 
    const int NO_PROCESSORS) {

    std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters);
    while (merges.size() != 0) {
        update_cluster_dissimilarities(merges, clusters, NO_PROCESSORS);

        update_cluster_nn(clusters, max_merge_distance);

        remove_secondary_clusters(merges, clusters);

        merges = find_reciprocal_nn(clusters);
    }
}

void RAC_i(
    std::vector<Cluster*>& clusters, 
    double max_merge_distance, 
    Eigen::MatrixXd& base_arr,
    const int NO_PROCESSORS) {

    std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters);
    while (merges.size() != 0) {
        update_cluster_dissimilarities(merges, clusters, NO_PROCESSORS, base_arr);

        update_cluster_nn(clusters, max_merge_distance);

        remove_secondary_clusters(merges, clusters);

        merges = find_reciprocal_nn(clusters);
    }
}

void RAC_i(
    std::vector<Cluster*>& clusters, 
    double max_merge_distance,
    const int NO_PROCESSORS,
    Eigen::MatrixXd& distance_arr) {

    std::vector<std::pair<int, int>> merges = find_reciprocal_nn(clusters);
    while (merges.size() != 0) {
        update_cluster_dissimilarities(merges, clusters, distance_arr, max_merge_distance, NO_PROCESSORS);

        update_cluster_nn(clusters, distance_arr, max_merge_distance);

        remove_secondary_clusters(merges, clusters);

        merges = find_reciprocal_nn(clusters);
    }
}

std::vector<int> RAC(
    Eigen::MatrixXd& base_arr,
    double max_merge_distance,
    Eigen::SparseMatrix<bool>& connectivity,
    std::string connectivity_type,
    int batch_size = 0,
    int no_processors = 0) {

    if (connectivity_type.empty()) {
        connectivity_type = "full";
    }

    //Processor Count defaults to the number on the machine if not provided or -1 passed
    const int NO_PROCESSORS = (no_processors != 0) ? no_processors : getProcessorCount();

    //Collect number of points in base_arr for space allocation
    const int NO_POINTS = base_arr.rows();

    //Batch Size defaults to NO_POINTS / 10 if not provided or -1 passed
    const int BATCHSIZE = (batch_size != 0) ? batch_size : NO_POINTS / 10; 

    base_arr = base_arr.transpose().colwise().normalized().eval();

    Eigen::setNbThreads(NO_PROCESSORS);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Cluster*> clusters;
    for (long i = 0; i < base_arr.cols(); ++i) {
        Cluster* cluster = new Cluster(i);
        clusters.push_back(cluster);
    }
    Eigen::MatrixXd distance_arr;

    if (connectivity.size() != 0) {
        calculate_initial_dissimilarities(base_arr, clusters, connectivity, max_merge_distance, BATCHSIZE);
    } else {
        distance_arr = calculate_initial_dissimilarities(base_arr, clusters, max_merge_distance);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "Time taken to calculate initial dissimilarities: " << duration << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    if (connectivity_type == "full") {
        RAC_i(clusters, max_merge_distance, NO_PROCESSORS, distance_arr);
    } else if (connectivity_type == "compute") {
        RAC_i(clusters, max_merge_distance, NO_PROCESSORS, base_arr); 
    } else { // Connctivity_type is "symmetric"
        RAC_i(clusters, max_merge_distance, NO_PROCESSORS); 
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout << "Time taken to calculate RAC: " << duration << "ms" << std::endl;

    // Set Eigen Threads according to Number of processors
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


//------------------------PYBIND INTERFACE----------------------------------

//Wrapper for RAC, convert return vector to a numpy array
py::array RAC_py(
    Eigen::MatrixXd base_arr,
    double max_merge_distance,
    Eigen::SparseMatrix<bool> connectivity,
    std::string connectivity_type,
    int batch_size = 0,
    int no_processors = 0) {

    std::vector<int> cluster_labels = RAC(base_arr, max_merge_distance, connectivity, connectivity_type, batch_size, no_processors);

    py::array cluster_labels_arr =  py::cast(cluster_labels); //py::array(vect_arr.size(), vect_arr.data());
    return cluster_labels_arr;
}

void simple_pybind_io_test() {
    std::cout << std::endl;
    std::cout << "This is a simple pybind I/O Test." << std::endl;
    std::cout << std::endl;
}

PYBIND11_MODULE(_racplusplus, m){
    m.doc() = R"doc( 
        RACplusplus is a C++ optimized python package for performing
        reciprocal agglomerative clustering.

        Authors: Porter Hunley, Daniel Frees
        2023
    )doc";

    m.def("rac", &RAC_py, R"fdoc(
        Run RAC algorithm on a provided array of points.

        Params:
        [base_arr] -        Actual data points array to be clustered. Each row is a point, with each column
                            representing the points value for a particular feature/dimension.
        [max_merge_distance] - Hyperparameter, maximum distance allowed for two clusters to merge with one another.
        [batch_size] -      Optional hyperparameter, batch size for calculating initial dissimilarities 
                            with a connectivity matrix.
                            Default: Defaults to the number of points in base_arr / 10 if 0 passed or no value passed.
        [connectivity] -    Optional: Connectivity matrix indicating whether points can be considered as neighbors.
                            Value of 1 at index i,j indicates point i and j are connected, 0 indicates disconnected.
                            Default: No connectivity matrix, use pairwise cosine to calculate distances.
        [connectivity_type]-Optional: Connectivity type indicates which compute method is used when connectivity is provided.
                            Options: "full" - Assume everything is connected (Default).
                                     "compute" - Use a hybrid model by which missing links are computed and cached on the fly. More
                                     balanced but at computational cost. 
                                     "symmetric" - Use a weighted average of all connections between two clusters 
                                     to compute the distance between them.
                            Default: "full"
        [no_processors] -   Hyperparameter, number of processors to use during computation. 
                            Defaults to the number of processors found on your machine if 0 passed 
                            or no value passed.

        Output:
        Returns a numpy array of the group # each point in base_arr was assigned to.
    )fdoc");

    m.def("test_rac", &main, R"fdoc(
        Testing function to run and time RAC's run in C++.
    )fdoc");

    m.def("simple_pybind_io_test", &simple_pybind_io_test, R"fdoc(
        Simple test function to see if pybind works, and can print text in python.
    )fdoc");

    m.attr("__version__") = "0.9";
}
//------------------------END PYBIND INTERFACE----------------------------------