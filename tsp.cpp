#pragma GCC optimize("O3")

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <limits>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <chrono>  // For timing
#include <iomanip> // For formatting time
#include <omp.h>   // Include OpenMP header
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <atomic>  // For atomic flags
#include <mutex>   // For thread-safe operations
#include <csignal> // For signal handling

using namespace std;
namespace fs = std::filesystem;

// **Global Atomic Flags for Interrupt and Population Failure**
std::atomic<bool> interrupted(false);
std::atomic<bool> population_failure(false);

// **Signal Handler Function**
void signal_handler(int signum)
{
    if (signum == SIGINT)
    {
        interrupted.store(true);
        // Note: Avoid using non-async-signal-safe functions like cout in signal handlers.
    }
}

// Custom hash function for vector<int>
struct VectorHash
{
    std::size_t operator()(const std::vector<int> &v) const
    {
        std::size_t seed = v.size();
        for (auto &i : v)
        {
            seed ^= std::hash<int>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

std::string format_time(double seconds)
{
    int hrs = static_cast<int>(seconds) / 3600;
    int mins = (static_cast<int>(seconds) % 3600) / 60;
    int secs = static_cast<int>(seconds) % 60;
    std::ostringstream oss;
    oss << std::setw(2) << std::setfill('0') << hrs << ":"
        << std::setw(2) << std::setfill('0') << mins << ":"
        << std::setw(2) << std::setfill('0') << secs;
    return oss.str();
}

// Function to check if a route is unique within a set
bool is_unique(const std::vector<int> &route, const std::unordered_set<std::vector<int>, VectorHash> &population_set)
{
    return population_set.find(route) == population_set.end();
}

struct Customers
{
    int name;
    double x, y;
};

struct result
{
    vector<int> path;
    double lowest_distance, min_distance_before_reset;
};

struct ProblemData
{
    int customers_count;
    vector<Customers> customers;
};
ProblemData readProblemData(const string &problem_name)
{
    ProblemData data;

    // Define ROOT as the current directory (can be adjusted as needed)
    fs::path ROOT = fs::current_path();

    // Create the path to the data file
    fs::path data_file = ROOT / "problems" / "tsp" / (problem_name + ".tsp") / (problem_name + ".tsp");

    ifstream infile(data_file);
    if (!infile.is_open())
    {
        cerr << "Error: Cannot open file " << data_file << endl;
        exit(EXIT_FAILURE);
    }

    string line;

    // Read general parameters
    while (getline(infile, line))
    {
        if (line.empty())
            continue; // Skip empty lines
        else if (line.find("NAME") != string::npos)
            continue;
        else if (line.find("COMMENT") != string::npos)
            continue;
        else if (line.find("TYPE") != string::npos)
            continue;
        else if (line.find("DIMENSION") != string::npos)
        {
            continue;
        }
        else if (line.find("EDGE_WEIGHT_TYPE"))
            break;
    }
    Customers depot;
    depot.x = 0.0;
    depot.y = 0.0;
    data.customers.push_back(depot);

    getline(infile, line); // Read the header line

    while (getline(infile, line))
    {
        if (line.empty())
            continue;
        Customers customer;
        stringstream ss(line);
        ss >> customer.name >> customer.x >> customer.y;
    }
    infile.close();

    data.customers_count += 1;

    return data;
}
double distance(const Customers &c1, const Customers &c2)
{
    double dx = c1.x - c2.x;
    double dy = c1.y - c2.y;
    return sqrt(dx * dx + dy * dy);
}

double total_distance(const vector<int> &route, const vector<Customers> &customers)
{
    double total = 0.0;
    for (size_t i = 0; i < route.size() - 1; ++i)
    {
        total += distance(customers[route[i]], customers[route[i + 1]]);
    }
    // Add distance from the last city back to depot if not already included
    if (route.back() != 0)
    {
        total += distance(customers[route.back()], customers[0]);
    }
    return total;
}
// Function to find forbidden transitions
unordered_map<int, unordered_set<int>> find_long_distance_pairs(const vector<Customers> &customers)
{
    double max_distance = 0.0;
    int num_customers = customers.size();
    // Find the maximum distance
    for (int i = 0; i < num_customers; ++i)
    {
        for (int j = i + 1; j < num_customers; ++j)
        {
            double d = distance(customers[i], customers[j]);
            if (d > max_distance)
                max_distance = d;
        }
    }

    // Calculate threshold
    double threshold = (63.0 / 100.0) * max_distance;

    // Collect forbidden transitions
    unordered_map<int, unordered_set<int>> forbidden_transitions;
    for (int i = 0; i < num_customers; ++i)
    {
        for (int j = 0; j < num_customers; ++j)
        {
            if (i == j)
                continue;
            double d = distance(customers[i], customers[j]);
            if (d > threshold)
            {
                forbidden_transitions[i].insert(j);
            }
        }
    }

    return forbidden_transitions;
}

// Function to check if a route contains any forbidden transitions
bool contains_forbidden_transition(const vector<int> &route, const unordered_map<int, unordered_set<int>> &forbidden_transitions)
{
    for (size_t i = 0; i < route.size() - 1; ++i)
    {
        int from = route[i];
        int to = route[i + 1];
        if (forbidden_transitions.find(from) != forbidden_transitions.end())
        {
            if (forbidden_transitions.at(from).find(to) != forbidden_transitions.at(from).end())
            {
                return true; // Contains forbidden transition
            }
        }
    }
    return false; // No forbidden transitions
}

// Function to create the initial population with unique routes, starting with depot (0)
vector<vector<int>> create_population(int pop_size, int num_customers, const unordered_map<int, unordered_set<int>> &forbidden_transitions)
{
    vector<vector<int>> population;
    std::unordered_set<std::vector<int>, VectorHash> population_set;

    // Create a list of cities (customers) from 1 to num_customers - 1
    vector<int> remaining_cities;
    for (int i = 1; i < num_customers; ++i)
    {
        remaining_cities.push_back(i);
    }

    // Initialize random number generator
    random_device rd;
    mt19937 gen(rd());

    // **Start Timer for Time-Based Stopping**
    auto start_time = chrono::steady_clock::now();
    const double time_limit = 20.0; // 20 seconds

    while (population.size() < static_cast<size_t>(pop_size))
    {
        // **Check for Time Limit Exceeded**
        auto current_time = chrono::steady_clock::now();
        chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() > time_limit)
        {
            population_failure.store(true);
            cerr << "\nError: Time limit exceeded while generating the initial population after " << elapsed.count() << " seconds.\n";
            break;
        }

        // Create a copy of the remaining cities
        vector<int> route = remaining_cities;
        // Shuffle the remaining cities randomly
        shuffle(route.begin(), route.end(), gen);
        // Add depot (0) at the beginning
        route.insert(route.begin(), 0);

        // Check if route contains forbidden transitions
        if (!contains_forbidden_transition(route, forbidden_transitions))
        {
            // Check for uniqueness
            if (is_unique(route, population_set))
            {
                population.push_back(route);
                population_set.insert(route);
            }
        }
    }

    if (population.size() < static_cast<size_t>(pop_size))
    {
        if (!population_failure.load())
        {
            population_failure.store(true);
            cerr << "\nError: Unable to generate the initial population within the time limit of " << time_limit << " seconds.\n";
        }
    }

    return population;
}

// Function to calculate fitness of a route
double fitness(const vector<int> &route, const vector<Customers> &customers)
{
    double total = total_distance(route, customers);
    // Avoid division by zero
    if (total == 0.0)
        return numeric_limits<double>::max();
    return 1.0 / total;
}

// Function to select a parent completely at random
vector<int> selection_random(const vector<vector<int>> &population)
{
    // Initialize random number generator (Thread-Local)
    thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> dis(0, population.size() - 1);

    // Select a random index
    int random_index = dis(gen);

    return population[random_index];
}

// Function to perform crossover between two parents to create a child, ensuring depot (0) remains at the start
vector<int> crossover(const vector<int> &parent1, const vector<int> &parent2)
{
    int size = parent1.size();
    // Initialize random number generator (Thread-Local)
    thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> dis(1, size - 2); // Avoid depot and last position

    int start = dis(gen);
    int end = dis(gen);

    // Ensure start < end
    if (start > end)
        swap(start, end);

    // Initialize child with -1 for other positions
    vector<int> child(size, -1);
    child[0] = 0; // Set depot at the start

    // Copy the middle segment from parent1
    for (int i = start; i <= end; ++i)
    {
        child[i] = parent1[i];
    }

    // Traverse parent2 and add cities not already in the child
    int pointer = 1; // Start from 1 because depot is already set
    for (int city : parent2)
    {
        if (city == 0)
            continue; // Skip depot
        if (find(child.begin(), child.end(), city) == child.end())
        {
            // Find the next available position
            while (pointer < size && child[pointer] != -1)
            {
                pointer++;
            }
            if (pointer < size)
            {
                child[pointer] = city;
                pointer++;
            }
        }
    }

    // Replace any remaining -1 with depot to ensure a valid route
    for (size_t i = 1; i < child.size(); ++i)
    {
        if (child[i] == -1)
            child[i] = 0;
    }

    return child;
}

// **New Mutation Function: Segment Swap Mutation**
// This function selects four distinct points in the route (excluding depot) and swaps the two segments defined by these points
void mutate_swap_segments(vector<int> &route)
{
    // Initialize random number generator (Thread-Local)
    thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> dis(1, route.size() - 1); // exclude depot

    // Select 4 distinct positions
    int pos1 = dis(gen);
    int pos2 = dis(gen);
    while (pos2 == pos1)
        pos2 = dis(gen);
    int pos3 = dis(gen);
    while (pos3 == pos1 || pos3 == pos2)
        pos3 = dis(gen);
    int pos4 = dis(gen);
    while (pos4 == pos1 || pos4 == pos2 || pos4 == pos3)
        pos4 = dis(gen);

    // Sort the positions
    vector<int> positions = {pos1, pos2, pos3, pos4};
    sort(positions.begin(), positions.end());

    // Define segments
    int p1 = positions[0];
    int p2 = positions[1];
    int p3 = positions[2];
    int p4 = positions[3];

    // Ensure that the segments do not overlap
    if (p2 >= p3)
    {
        // Overlapping segments, abort mutation
        return;
    }

    // Extract segments
    vector<int> segment1(route.begin() + p1, route.begin() + p2 + 1); // inclusive
    vector<int> segment2(route.begin() + p3, route.begin() + p4 + 1); // inclusive

    // Remove segment2 first to prevent shifting indices
    route.erase(route.begin() + p3, route.begin() + p4 + 1);
    route.erase(route.begin() + p1, route.begin() + p2 + 1);

    // Insert segment2 at p1
    route.insert(route.begin() + p1, segment2.begin(), segment2.end());

    // Insert segment1 at the new position (p3 shifted by the size of segment2)
    int new_p3 = p1 + segment2.size();
    route.insert(route.begin() + new_p3, segment1.begin(), segment1.end());
}

// Function to perform segment reversal mutation
void mutate_reverse_segment(vector<int> &route)
{
    // Initialize random number generator (Thread-Local)
    thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> dis_length(1, route.size() - 2); // Exclude depot

    // Determine the length of the segment to reverse (10% to 40% of the route length)
    int total_segment_length = route.size() - 1; // Exclude depot
    int min_segment_length = static_cast<int>(0.1 * total_segment_length);
    int max_segment_length = static_cast<int>(0.4 * total_segment_length);

    // Ensure that min_segment_length is at least 2 to allow meaningful reversal
    min_segment_length = max(2, min_segment_length);
    max_segment_length = max(min_segment_length, max_segment_length); // Ensure max >= min

    // Uniform distribution for segment length within the specified range
    uniform_int_distribution<> dis_segment_length(min_segment_length, max_segment_length);
    int segment_length = dis_segment_length(gen);

    // Select a random start position ensuring the segment fits within the route
    uniform_int_distribution<> dis_start(1, route.size() - 1 - segment_length); // Exclude depot
    int start_pos = dis_start(gen);
    int end_pos = start_pos + segment_length - 1;

    // Reverse the segment [start_pos, end_pos]
    reverse(route.begin() + start_pos, route.begin() + end_pos + 1);
}

// **Updated Mutation Function: Includes Segment Reversal and Segment Swap Mutations**
// Mutation types:
// - 35% Probability: Segment Reversal
// - 65% Probability: Segment Swap
void mutate(vector<int> &route, double mutation_rate)
{
    // Initialize random number generator (Thread-Local)
    thread_local mt19937 gen(random_device{}());
    uniform_real_distribution<> dis_real(0.0, 1.0);
    uniform_real_distribution<> dis_choice(0.0, 1.0);

    // Decide whether to perform mutation based on mutation_rate
    if (dis_real(gen) < mutation_rate)
    {
        // Decide mutation type based on defined probabilities
        double choice = dis_choice(gen);
        if (choice < 0.35) // 35% chance for segment reversal mutation
        {
            // Perform segment reversal mutation
            mutate_reverse_segment(route);
        }
        else // 65% chance for segment swap mutation
        {
            // Perform segment swap mutation
            mutate_swap_segments(route);
        }
    }
}

// Function to remove the worst 50% of individuals and replace them with new unique individuals
void del50(vector<vector<int>> &population, const vector<double> &fitnesses, int pop_size, int num_customers, unordered_set<std::vector<int>, VectorHash> &population_set, const unordered_map<int, unordered_set<int>> &forbidden_transitions)
{
    // Determine number to remove (50% of population)
    int number_to_remove = pop_size / 2;

    // Create a vector of indices sorted by fitness ascending (worst fitness first)
    vector<int> indices(pop_size);
    for (int i = 0; i < pop_size; ++i)
    {
        indices[i] = i;
    }

    // Sort indices based on fitnesses (ascending order: worst fitness first)
    sort(indices.begin(), indices.end(), [&](int a, int b)
         {
             return fitnesses[a] < fitnesses[b]; // Ascending order: worst fitness first
         });

    // Collect the indices to remove
    vector<int> remove_indices(indices.begin(), indices.begin() + number_to_remove);

    // Collect the routes to remove
    vector<vector<int>> removed_routes;
    removed_routes.reserve(number_to_remove);
    for (int idx : remove_indices)
    {
        removed_routes.push_back(population[idx]);
    }

    // Remove the worst 50% from population and population_set
    // To remove efficiently, sort remove_indices in descending order
    sort(remove_indices.begin(), remove_indices.end(), greater<int>());
    for (int idx : remove_indices)
    {
        population_set.erase(population[idx]);
        population.erase(population.begin() + idx);
    }

    // Generate number_to_remove new unique individuals
    // Ensure they are not duplicates of removed_routes and not duplicates within new_population
    unordered_set<std::vector<int>, VectorHash> removed_set(removed_routes.begin(), removed_routes.end(), number_to_remove, VectorHash());

    int new_individuals_needed = number_to_remove;

    // Create a list of cities (customers) from 1 to num_customers - 1
    vector<int> remaining_cities;
    for (int i = 1; i < num_customers; ++i)
    {
        remaining_cities.push_back(i);
    }

    // Initialize random number generator
    random_device rd;
    mt19937 gen(rd());

    // **Start Timer for Time-Based Stopping**
    auto start_time = chrono::steady_clock::now();
    const double time_limit = 20.0; // 20 seconds

    while (new_individuals_needed > 0)
    {
        // **Check for Time Limit Exceeded**
        auto current_time = chrono::steady_clock::now();
        chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() > time_limit)
        {
            population_failure.store(true);
            cerr << "\nError: Time limit exceeded while replacing individuals in del50 after " << elapsed.count() << " seconds.\n";
            break;
        }

        // Create a copy of the remaining cities
        vector<int> route = remaining_cities;
        // Shuffle the remaining cities randomly
        shuffle(route.begin(), route.end(), gen);
        // Add depot (0) at the beginning
        route.insert(route.begin(), 0);

        // Check if route contains forbidden transitions
        if (!contains_forbidden_transition(route, forbidden_transitions))
        {
            // Check for uniqueness
            if (is_unique(route, population_set) && removed_set.find(route) == removed_set.end())
            {
                population.push_back(route);
                population_set.insert(route);
                new_individuals_needed--;
            }
        }
    }

    if (new_individuals_needed > 0)
    {
        if (!population_failure.load())
        {
            population_failure.store(true);
            cerr << "\nError: Unable to replace " << new_individuals_needed << " individuals within the time limit of " << time_limit << " seconds in del50.\n";
        }
    }
}

// Function to create a new population excluding specific sets
vector<vector<int>> create_new_population(int pop_size, int num_customers, const unordered_set<std::vector<int>, VectorHash> &exclude_set_initial, const unordered_set<std::vector<int>, VectorHash> &exclude_set_last, const unordered_map<int, unordered_set<int>> &forbidden_transitions)
{
    vector<vector<int>> new_population;
    std::unordered_set<std::vector<int>, VectorHash> population_set;

    // Combine the exclusion sets for quick lookup
    auto combined_exclude = [&](const vector<int> &route) -> bool
    {
        return exclude_set_initial.find(route) != exclude_set_initial.end() ||
               exclude_set_last.find(route) != exclude_set_last.end();
    };

    // Create a list of cities (customers) from 1 to num_customers - 1
    vector<int> remaining_cities;
    for (int i = 1; i < num_customers; ++i)
    {
        remaining_cities.push_back(i);
    }

    // Initialize random number generator
    random_device rd;
    mt19937 gen(rd());

    // **Start Timer for Time-Based Stopping**
    auto start_time = chrono::steady_clock::now();
    const double time_limit = 20.0; // 20 seconds

    while (new_population.size() < static_cast<size_t>(pop_size))
    {
        // **Check for Time Limit Exceeded**
        auto current_time = chrono::steady_clock::now();
        chrono::duration<double> elapsed = current_time - start_time;
        if (elapsed.count() > time_limit)
        {
            population_failure.store(true);
            cerr << "\nError: Time limit exceeded while creating a new population after " << elapsed.count() << " seconds.\n";
            break;
        }

        // Create a copy of the remaining cities
        vector<int> route = remaining_cities;
        // Shuffle the remaining cities randomly
        shuffle(route.begin(), route.end(), gen);
        // Add depot (0) at the beginning
        route.insert(route.begin(), 0);

        // Check if route contains forbidden transitions
        if (!contains_forbidden_transition(route, forbidden_transitions))
        {
            // Check for uniqueness against exclusion sets
            if (!combined_exclude(route))
            {
                new_population.push_back(route);
                population_set.insert(route);
            }
        }
    }

    if (new_population.size() < static_cast<size_t>(pop_size))
    {
        if (!population_failure.load())
        {
            population_failure.store(true);
            cerr << "\nError: Unable to create a complete new population within the time limit of " << time_limit << " seconds.\n";
        }
    }

    return new_population;
}

// GA algorithm to find the shortest route
result solveGA(const ProblemData &data, int pop_size = 50, int generations = 1000, double mutation_rate = 0.1)
{
    int num_customers = data.customers_count; // Now includes depot
    const vector<Customers> &customers = data.customers;

    // **Find Forbidden Transitions**
    unordered_map<int, unordered_set<int>> forbidden_transitions = find_long_distance_pairs(customers);
    cout << "Number of forbidden transitions: " << forbidden_transitions.size() << endl;
    /*
    for (auto &pair : forbidden_transitions)
    {
        // Count total forbidden transitions
        // Each entry in the map has a set of forbidden B for a given A
        // Uncomment below to see the count
        // cout << "From " << pair.first << " to " << pair.second.size() << " points" << endl;
    }
    */

    // **Create the Initial Population**
    vector<vector<int>> population = create_population(pop_size, num_customers, forbidden_transitions);
    // Create a set to track uniqueness (Specify bucket_count, e.g., pop_size * 2)
    unordered_set<std::vector<int>, VectorHash> population_set(population.begin(), population.end(), pop_size * 2, VectorHash());

    // Store the initial population for future exclusion
    unordered_set<std::vector<int>, VectorHash> initial_population_set(population.begin(), population.end(), pop_size * 2, VectorHash());

    // **Check for Population Creation Failure**
    if (population_failure.load())
    {
        cerr << "Initial population creation failed. Exiting GA.\n";
        // Proceed to find the best individual in the (possibly partial) population
    }

    // Variables to track improvement
    double best_distance = numeric_limits<double>::max();
    vector<int> best_route;
    int generations_without_improvement = 0;
    const int max_generations_without_improvement = 50;

    // List to store best solutions before each reset
    vector<pair<vector<int>, double>> best_solutions_before_reset;

    // Initialize current mutation rate
    double current_mutation_rate = mutation_rate;

    // Initialize del50 counter
    int del50_counter = 0;

    // Initialize reset counter
    int reset_counter = 0;

    // Start timing
    auto start_time = chrono::steady_clock::now();

    for (int generation = 0; generation < generations; ++generation)
    {
        // **Check for Interruption**
        if (interrupted.load())
        {
            cout << "\nInterrupt signal received. Terminating GA early.\n";
            break;
        }

        // **Check for Population Failure**
        if (population_failure.load())
        {
            cout << "\nPopulation failure detected. Terminating GA early.\n";
            break;
        }

        // Calculate fitness for each individual (Parallelized with OpenMP)
        vector<double> fitnesses(population.size());

#pragma omp parallel for
        for (int i = 0; i < population.size(); ++i)
        {
            fitnesses[i] = fitness(population[i], customers);
        }

        // Find the best individual in the current generation (Parallelized with OpenMP)
        double current_best_distance = numeric_limits<double>::max();
        int best_index = -1;

#pragma omp parallel for
        for (int i = 0; i < population.size(); ++i)
        {
            double current_distance = total_distance(population[i], customers);
            // Use a critical section to safely update the best distance and index
#pragma omp critical
            {
                if (current_distance < current_best_distance)
                {
                    current_best_distance = current_distance;
                    best_index = i;
                }
            }
        }

        // If a new best route was found, update current_best_route
        vector<int> current_best_route;
        if (best_index != -1)
        {
            current_best_route = population[best_index];
        }

        // Check and update the best result
        if (current_best_distance < best_distance)
        {
            best_distance = current_best_distance;
            best_route = current_best_route;
            generations_without_improvement = 0; // Reset the counter
            // Optionally, store or print the improvement
            // cout << "Generation " << generation << ": Best Distance = " << best_distance << endl;
        }
        else
        {
            generations_without_improvement++;
            // If no improvement after max_generations_without_improvement generations, apply del50
            if (generations_without_improvement >= max_generations_without_improvement)
            {
                // Store the best solution before reset
                if (!best_route.empty())
                {
                    best_solutions_before_reset.emplace_back(best_route, best_distance);
                }

                // Inform about applying del50
                cout << "\nNo improvement after " << max_generations_without_improvement << " generations. Applying del50 to rejuvenate population..." << endl;

                // Apply del50 to remove worst 50% and replace with new unique individuals
                del50(population, fitnesses, pop_size, num_customers, population_set, forbidden_transitions);
                del50_counter++;

                // **Check for Population Failure After del50**
                if (population_failure.load())
                {
                    cout << "Population replacement failed during del50. Terminating GA early.\n";
                    break;
                }

                // Check if del50 has been called 5 times for complete reset
                if (del50_counter >= 5)
                {
                    // Perform a complete reset
                    reset_counter++;
                    del50_counter = 0; // Reset the del50 counter

                    // Store the current population before reset
                    unordered_set<std::vector<int>, VectorHash> last_population_before_reset(population.begin(), population.end(), pop_size * 2, VectorHash());

                    // Inform about the complete reset
                    cout << "Del50 has been invoked 5 times. Performing a complete population reset..." << endl;

                    // Generate a completely new population excluding initial and last populations
                    population = create_new_population(pop_size, num_customers, initial_population_set, last_population_before_reset, forbidden_transitions);

                    // Update the population_set with the new population
                    population_set.clear();
                    for (const auto &route : population)
                    {
                        population_set.insert(route);
                    }

                    // **Check for Population Failure After Complete Reset**
                    if (population_failure.load())
                    {
                        cout << "Population reset failed. Terminating GA early.\n";
                        break;
                    }

                    // Reset the mutation rate to 0.2 as per the requirement
                    current_mutation_rate = 0.2;
                    cout << "Mutation rate has been reset to: " << current_mutation_rate << endl;

                    // Inform about the reset
                    cout << "Complete population reset completed. Generation " << generation + 1 << " continues." << endl;
                }
                else
                {
                    // Increase mutation rate by 1.3x, cap at 1.0
                    current_mutation_rate *= 1.3;
                    if (current_mutation_rate > 1.0)
                        current_mutation_rate = 1.0;

                    // Inform about the increased mutation rate
                    cout << "Mutation rate increased to: " << current_mutation_rate << endl;
                }

                generations_without_improvement = 0; // Reset the counter
            }
        }

        // **Genetic Operations: Selection, Crossover, Mutation**

        vector<vector<int>> new_population_genetic;
        unordered_set<std::vector<int>, VectorHash> new_population_set_genetic;

        // To ensure uniqueness, we'll track new individuals in new_population_set
        // Define maximum attempts to prevent infinite loops
        // **Start Timer for Time-Based Stopping**
        auto genetic_start_time = chrono::steady_clock::now();
        const double genetic_time_limit = 20.0; // 20 seconds

        while (new_population_genetic.size() < static_cast<size_t>(pop_size))
        {
            // **Check for Time Limit Exceeded**
            auto current_time = chrono::steady_clock::now();
            chrono::duration<double> elapsed_genetic = current_time - genetic_start_time;
            if (elapsed_genetic.count() > genetic_time_limit)
            {
                population_failure.store(true);
                cerr << "\nError: Time limit exceeded while generating new population during genetic operations after " << elapsed_genetic.count() << " seconds.\n";
                break;
            }

            // **Check for Interruption During Genetic Operations**
            if (interrupted.load())
            {
                cout << "\nInterrupt signal received during genetic operations. Terminating GA early.\n";
                break;
            }

            // Select two parents randomly
            vector<int> parent1 = selection_random(population);
            vector<int> parent2 = selection_random(population);

            // Perform crossover to produce a child
            vector<int> child = crossover(parent1, parent2);

            // Perform mutation on the child with the current mutation rate
            mutate(child, current_mutation_rate); // Updated to include only two mutation types

            // Ensure the child does not contain forbidden transitions
            if (!contains_forbidden_transition(child, forbidden_transitions))
            {
                // Ensure the child is unique before adding
                if (is_unique(child, population_set) && is_unique(child, new_population_set_genetic))
                {
                    new_population_genetic.push_back(child);
                    new_population_set_genetic.insert(child);
                }
            }
        }

        // **Check for Population Insufficiency During Genetic Operations**
        if (new_population_genetic.size() < static_cast<size_t>(pop_size))
        {
            population_failure.store(true);
            cerr << "\nError: Unable to generate a complete new population during genetic operations within the time limit of " << genetic_time_limit << " seconds.\n";
        }

        // **Replace the Old Population with the New Population Only if No Interruption or Failure**
        if (!interrupted.load() && !population_failure.load())
        {
            population = new_population_genetic;

            // Update the population_set with the new population
            population_set.clear();
            for (const auto &route : population)
            {
                population_set.insert(route);
            }
        }
        else
        {
            // If interrupted or population failure, skip replacing the population
            // Proceed to finalize the GA
            if (interrupted.load())
                cout << "\nGA terminated due to user interruption.\n";
            if (population_failure.load())
                cout << "\nGA terminated due to population insufficiency.\n";
            break;
        }

        // Update progress bar every generation
        // Calculate elapsed time
        auto current_time_loop = chrono::steady_clock::now();
        chrono::duration<double> elapsed = current_time_loop - start_time;
        double elapsed_seconds = elapsed.count();

        // Calculate estimated total time
        double progress = static_cast<double>(generation + 1) / generations;
        double estimated_total_seconds = (progress > 0.0) ? (elapsed_seconds / progress) : 0.0;
        double remaining_seconds = estimated_total_seconds - elapsed_seconds;

        // Format time
        string elapsed_str = format_time(elapsed_seconds);
        string remaining_str = format_time(remaining_seconds);

        // Create progress bar
        int bar_width = 50;
        int pos = static_cast<int>(bar_width * progress);
        cout << "\r[";
        for (int i = 0; i < bar_width; ++i)
        {
            if (i < pos)
                cout << "#";
            else
                cout << "-";
        }
        cout << "] " << fixed << setprecision(2) << (progress * 100.0) << "% "
             << "Generations: " << (generation + 1) << "/" << generations << " "
             << "Elapsed Time: " << elapsed_str << " "
             << "Remaining Time: " << remaining_str
             << std::flush;
    }

    // Find the best individual in the final population (if not already found)
    if (best_route.empty())
    {
        double min_distance = numeric_limits<double>::max();
        int best_index = -1;
#pragma omp parallel for
        for (int i = 0; i < population.size(); ++i)
        {
            double current_distance = total_distance(population[i], customers);
            // Use a critical section to safely update the best distance and index
#pragma omp critical
            {
                if (current_distance < min_distance)
                {
                    min_distance = current_distance;
                    best_index = i;
                }
            }
        }
        if (best_index != -1)
        {
            best_route = population[best_index];
            best_distance = min_distance;
        }
    }

    // Store the best solution if not already stored
    if (!best_route.empty())
    {
        best_solutions_before_reset.emplace_back(best_route, best_distance);
    }

    // Find the minimum distance from best_solutions_before_reset
    double min_distance_before_reset = numeric_limits<double>::max();
    for (const auto &solution : best_solutions_before_reset)
    {
        if (solution.second < min_distance_before_reset)
        {
            min_distance_before_reset = solution.second;
        }
    }

    // Print the list of best solutions before each reset
    cout << "\n\nList of Best Solutions Found Before Each Reset:" << endl;
    for (size_t i = 0; i < best_solutions_before_reset.size(); ++i)
    {
        cout << "Best Solution " << i + 1 << ": Distance = " << best_solutions_before_reset[i].second << " | Route: ";
        for (size_t j = 0; j < best_solutions_before_reset[i].first.size(); ++j)
        {
            cout << best_solutions_before_reset[i].first[j];
            if (j != best_solutions_before_reset[i].first.size() - 1)
                cout << " -> ";
        }
        cout << endl;
    }

    // Print the minimum distance from the list
    cout << "\nMinimum distance from best_solutions_before_reset: " << min_distance_before_reset << endl;

    // Ensure the route ends at depot (0)
    if (best_route.back() != 0)
    {
        best_route.push_back(0);
        best_distance += distance(customers[best_route[best_route.size() - 2]], customers[0]);
    }

    // Populate the GAResult struct
    result result;
    result.path = best_route;
    result.lowest_distance = best_distance;
    result.min_distance_before_reset = min_distance_before_reset;

    return result;
}

int main(int argc, char *argv[])
{
    // **Register Signal Handler**
    signal(SIGINT, signal_handler);

    // Check the number of OpenMP threads being used
#pragma omp parallel
    {
#pragma omp single
        {
            cout << "Number of OpenMP threads: " << omp_get_num_threads() << endl;
        }
    }

    // Check command-line arguments
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <problem_name>" << endl;
        return EXIT_FAILURE;
    }

    string problem_name = argv[1];

    // Read data from file
    ProblemData data = readProblemData(problem_name);

    // Optionally, print customer information
    /*
    for (size_t i = 0; i < data.customers.size(); ++i) {
        const auto& customer = data.customers[i];
        cout << "Customer " << i << " - X: " << customer.x << ", Y: " << customer.y
             << ", Demand: " << customer.demand
             << ", OnlyServicedByStaff: " << customer.onlyServicedByStaff
             << ", ServiceTimeByTruck(s): " << customer.serviceTimeByTruck
             << ", ServiceTimeByDrone(s): " << customer.serviceTimeByDrone << endl;
    }
    */

    // Run the GA algorithm
    int population_size = 50;   // Can be adjusted
    int generations = 1000;     // Can be adjusted
    double mutation_rate = 0.1; // Initial mutation rate

    result result = solveGA(data, population_size, generations, mutation_rate);

    // Print the results
    cout << "\nBest distance: " << fixed << setprecision(2) << result.min_distance_before_reset << endl;
    cout << "Best route: ";
    for (size_t i = 0; i < result.path.size(); ++i)
    {
        cout << result.path[i];
        if (i != result.path.size() - 1)
            cout << " -> ";
    }
    cout << endl;

    // **Detailed Move Distance Reporting**
    cout << "\nDetailed Move Distances:" << endl;
    double total_distance_traveled = 0.0;
    for (size_t i = 0; i < result.path.size() - 1; ++i)
    {
        int from = result.path[i];
        int to = result.path[i + 1];
        double d = distance(data.customers[from], data.customers[to]);
        cout << "Move from point " << from << " to point " << to << " | Distance: " << fixed << setprecision(2) << d << endl;
        total_distance_traveled += d;
    }
    cout << "Total Distance Traveled: " << fixed << setprecision(2) << total_distance_traveled << endl;

    return 0;
}
