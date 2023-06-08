#include <array>
#include <tuple>
#include <unordered_map>
#include <set>
#include "Eigen/Dense"
#include "Eigen/Sparse"

#ifndef GLOBAL_TIMING_VARS_H
#define GLOBAL_TIMING_VARS_H

// Store update neighbor times
std::vector<long> UPDATE_NEIGHBOR_DURATIONS;
// Store update NN times
std::vector<long> UPDATE_NN_DURATIONS;
// Store the durations of each call to cosine
std::vector<long> COSINE_DURATIONS;
std::vector<long> INDICES_DURATIONS;
std::vector<long> MERGE_DURATIONS;
std::vector<long> MISC_MERGE_DURATIONS;

#endif // GLOBAL_TIMING_VARS_H

#ifndef CLUSTER_H 
#define CLUSTER_H

class Cluster {
public:
    int id;
    bool will_merge;
    int nn;
    std::vector<int> neighbors;
    std::vector<int> indices;
    std::unordered_map<int, float> dissimilarities;
    std::vector<std::tuple<int, int, double> > neighbors_needing_updates;
    
    Cluster(int id);

    void update_nn(float min_distance);
    void update_nn(Eigen::MatrixXd& distance_arr);
};

#endif //CLUSTER_H

//--------------------Helpers------------------------------------
//Function to optimize to # of processors
size_t getProcessorCount();

// Function to generate a matrix filled with random numbers.
Eigen::MatrixXd generateRandomMatrix(int rows, int cols, int seed);

double get_arr_value(Eigen::MatrixXd& arr, int i, int j);
void set_arr_value(Eigen::MatrixXd& arr, int i, int j, double value);

void remove_secondary_clusters(std::vector<std::pair<int, int> >& merges, std::vector<Cluster*>& clusters);
//--------------------End Helpers------------------------------------


//-----------------------Distance Calculations-------------------------
//Calculate pairwise cosines between two matrices
Eigen::MatrixXd pairwise_cosine(const Eigen::MatrixXd& array_a, const Eigen::MatrixXd& array_b);

//Averaged dissimilarity across two matrices (wrapper for pairwise distance calc + avging)
float calculate_weighted_dissimilarity(Eigen::MatrixXd points_a, Eigen::MatrixXd points_b);

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges, 
    std::vector<Cluster*>& clusters,
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>>& sort_neighbor,
    std::vector<std::vector<int>>& merging_arrays,
    const int NO_PROCESSORS);

void update_cluster_dissimilarities(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster*>& clusters,
    Eigen::MatrixXd& distance_arr,
    float min_distance,
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>>& sort_neighbor,
    std::vector<std::vector<int>>& merging_arrays,
    const int NO_PROCESSORS);

Eigen::MatrixXd calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster*>& clusters,
    float min_distance);

void calculate_initial_dissimilarities(
    Eigen::MatrixXd& base_arr,
    std::vector<Cluster*>& clusters,
    int batchSize,
    float min_distance,
    Eigen::SparseMatrix<bool>& connectivity);

//-----------------------End Distance Calculations-------------------------

//-----------------------Merging Functions-----------------------------------
void merge_cluster_apx(
    std::pair<int, int>& merge,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& distance_arr,
    float min_distance);

void merge_cluster_apx(
    std::pair<int, int>& merge,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array);

void merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array);

void merge_clusters_dist(
    std::vector<std::pair<int, int> >& merges,
    std::vector<Cluster*>& clusters,
    std::vector<int>& merging_array,
    Eigen::MatrixXd& distance_arr,
    float min_distance);

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges, 
    std::vector<Cluster*>& clusters,
    size_t no_threads,
    std::vector<std::vector<int>>& merging_arrays);

void parallel_merge_clusters(
    std::vector<std::pair<int, int> >& merges,
    Eigen::MatrixXd& distance_arr,
    std::vector<Cluster*>& clusters,
    size_t no_threads,
    float min_distance,
    std::vector<std::vector<int>>& merging_arrays);
//-----------------------End Merging Functions-----------------------------------

//-----------------------Updating Nearest Neighbors-----------------------------------

void update_cluster_neighbors(
    std::pair<int, std::vector<std::pair<int, double> > >& update_chunk,
    std::vector<Cluster*>& clusters,
    std::vector<int>& update_neighbors);

void update_cluster_neighbors(
    std::pair<int, std::vector<std::pair<int, double> > >& update_chunk,
    std::vector<Cluster*>& clusters,
    Eigen::MatrixXd& distance_arr,
    float min_distance,
    std::vector<int>& update_neighbors);

void update_cluster_neighbors_p(
    std::vector<std::pair<int, std::vector<std::pair<int, double> > > >& updates,
    std::vector<Cluster*>& clusters, 
    std::vector<int>& update_neighbors);

void parallel_update_clusters(
    std::vector<std::pair<int, std::vector<std::pair<int, double> > > >& updates,
    std::vector<Cluster*>& clusters,
    size_t no_threads, 
    std::vector<int>& update_neighbors);

void update_cluster_nn(
    std::vector<Cluster*>& clusters,
    Eigen::MatrixXd& distance_arr,
    float min_disitance);

std::vector<std::pair<int, int> > find_reciprocal_nn(std::vector<Cluster*>& clusters);
//-----------------------End Updating Nearest Neighbors-----------------------------------

//--------------------------------------RAC Functions--------------------------------------
void RAC_i(
    std::vector<Cluster*>& clusters, 
    float min_distance, 
    Eigen::MatrixXd& distance_arr, 
    std::vector<std::pair<int, std::vector<std::pair<int, double>>>>& sort_neighbor,
    std::vector<std::vector<int>>& merging_arrays,
    const int NO_PROCESSORS);

std::vector<int> RAC(
    Eigen::MatrixXd base_arr,
    float min_distance,
    int batchSize,
    Eigen::SparseMatrix<bool> connectivity,
    int no_processors,
    const int NO_POINTS);
//--------------------------------------End RAC Functions--------------------------------------

