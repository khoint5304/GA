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
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <atomic>
#include <mutex>
#include <csignal>

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
    double lowest_distance;
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
            // Extract the number of customers
            size_t pos = line.find(":");
            if (pos != string::npos)
            {
                data.customers_count = stoi(line.substr(pos + 1));
            }
        }
        else if (line.find("NODE_COORD_SECTION") != string::npos)
            break;
    }

    // Read customer coordinates
    while (getline(infile, line))
    {
        if (line.empty() || line.find("EOF") != string::npos)
            continue;
        Customers customer;
        stringstream ss(line);
        ss >> customer.name >> customer.x >> customer.y;
        data.customers.push_back(customer);
    }
    infile.close();

    return data;
}

double distance(int from, int to, const vector<Customers> &customers)
{
    double dx = customers[from - 1].x - customers[to - 1].x;
    double dy = customers[from - 1].y - customers[to - 1].y;
    return sqrt(dx * dx + dy * dy);
}

double total_distance(const vector<int> &route, const vector<Customers> &customers)
{
    double total = 0.0;
    for (size_t i = 0; i < route.size() - 1; ++i)
    {
        total += distance(route[i], route[i + 1], customers);
    }
    // Ensure the route ends at the depot (Customer 1)
    if (route.back() != 1)
    {
        total += distance(route.back(), 1, customers);
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
            double d = distance(i + 1, j + 1, customers);
            if (d > max_distance)
                max_distance = d;
        }
    }

    // Calculate threshold
    double threshold = (100.0 / 100.0) * max_distance; // Currently, 100% of max_distance

    // Collect forbidden transitions
    unordered_map<int, unordered_set<int>> forbidden_transitions;
    for (int i = 1; i <= num_customers; ++i)
    {
        for (int j = 1; j <= num_customers; ++j)
        {
            if (i == j)
                continue;
            double d = distance(i, j, customers);
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

// Function to create the initial population with unique routes, starting and ending with Customer 1
vector<vector<int>> create_population(int pop_size, int num_customers, const unordered_map<int, unordered_set<int>> &forbidden_transitions)
{
    vector<vector<int>> population;
    std::unordered_set<std::vector<int>, VectorHash> population_set;

    // Create a list of cities (customers) from 2 to num_customers
    vector<int> remaining_cities;
    for (int i = 2; i <= num_customers; ++i)
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
        // Add depot (1) at the beginning and end
        route.insert(route.begin(), 1);
        route.push_back(1);

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

// Function for tournament selection
vector<int> selection_tournament(const vector<vector<int>> &population, const vector<double> &fitnesses, int tournament_size = 5)
{
    thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> dis(0, population.size() - 1);

    double best_fitness = -1.0;
    int best_index = -1;

    for (int i = 0; i < tournament_size; ++i)
    {
        int idx = dis(gen);
        if (fitnesses[idx] > best_fitness)
        {
            best_fitness = fitnesses[idx];
            best_index = idx;
        }
    }

    return population[best_index];
}

// Function to perform Order Crossover (OX)
vector<int> crossover_order(const vector<int> &parent1, const vector<int> &parent2)
{
    int size = parent1.size();
    if (size <= 3)
        return parent1; // Cannot perform crossover, return one of the parents

    thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> dis(1, size - 2); // Exclude depot at start and end

    int start = dis(gen);
    int end = dis(gen);

    if (start > end)
        swap(start, end);

    // Initialize child with -1
    vector<int> child(size, -1);
    child[0] = 1;        // Depot
    child[size - 1] = 1; // Depot

    // Copy the slice from parent1 to child
    for (int i = start; i <= end; ++i)
    {
        child[i] = parent1[i];
    }

    // Fill the remaining positions with genes from parent2 in order
    int current = (end + 1) % (size - 1);
    int idx = 1; // Start from the first gene after the depot
    while (current != start)
    {
        if (child[current] == -1)
        {
            // Find the next gene from parent2 that is not already in child
            while (find(child.begin(), child.end(), parent2[idx]) != child.end())
            {
                idx = (idx + 1) % (size - 1);
                if (idx == 0)
                    idx = 1; // Skip depot
            }
            child[current] = parent2[idx];
            current = (current + 1) % (size - 1);
            idx = (idx + 1) % (size - 1);
            if (idx == 0)
                idx = 1; // Skip depot
        }
        else
        {
            current = (current + 1) % (size - 1);
        }
    }

    // Ensure that all positions are filled
    for (int i = 1; i < size - 1; ++i)
    {
        if (child[i] == -1)
        {
            // Fill with missing genes
            for (int gene = 2; gene <= size - 1; ++gene)
            {
                if (find(child.begin(), child.end(), gene) == child.end())
                {
                    child[i] = gene;
                    break;
                }
            }
        }
    }

    return child;
}

// Function to perform segment swap mutation
void mutate_swap_segments(vector<int> &route)
{
    if (route.size() < 5)
        return; // Not enough elements to perform mutation

    // Initialize random number generator (Thread-Local)
    thread_local mt19937 gen(random_device{}());
    uniform_int_distribution<> dis(1, route.size() - 2); // Exclude depot

    // Select 4 distinct positions
    unordered_set<int> positions_set;
    while (positions_set.size() < 4)
    {
        int pos = dis(gen);
        positions_set.insert(pos);
    }

    vector<int> positions(positions_set.begin(), positions_set.end());
    sort(positions.begin(), positions.end());

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
    if (route.size() < 4)
        return; // Not enough elements to perform mutation

    // Initialize random number generator (Thread-Local)
    thread_local mt19937 gen(random_device{}());

    int total_segment_length = route.size() - 2; // Exclude depot at both ends
    int min_segment_length = max(2, static_cast<int>(0.1 * total_segment_length));
    int max_segment_length = max(min_segment_length, static_cast<int>(0.4 * total_segment_length)); // Ensure max >= min

    if (total_segment_length < min_segment_length)
        return; // Segment length too long for the route

    // Uniform distribution for segment length within the specified range
    uniform_int_distribution<> dis_segment_length(min_segment_length, max_segment_length);
    int segment_length = dis_segment_length(gen);

    if (route.size() - 1 - segment_length < 1)
        return; // Cannot perform mutation, segment too long

    uniform_int_distribution<> dis_start(1, route.size() - 1 - segment_length); // Exclude depot
    int start_pos = dis_start(gen);
    int end_pos = start_pos + segment_length - 1;

    reverse(route.begin() + start_pos, route.begin() + end_pos + 1);
}

// **Forward Declaration of two_opt Function**
void two_opt(vector<int> &route, const vector<Customers> &customers, int max_iterations = 1000);

// Mutation function with segment reversal and swap
void mutate(vector<int> &route, double mutation_rate, const vector<Customers> &customers)
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

        // **Apply 2-opt for Local Optimization**
        two_opt(route, customers, 1000); // Limit iterations to prevent long runtimes
    }
}

// **Implement 2-opt Local Search**
void two_opt_swap(vector<int> &route, int i, int k)
{
    while (i < k)
    {
        swap(route[i], route[k]);
        i++;
        k--;
    }
}

void two_opt(vector<int> &route, const vector<Customers> &customers, int max_iterations)
{
    bool improvement = true;
    int size = route.size();
    int iterations = 0;
    while (improvement && iterations < max_iterations)
    {
        improvement = false;
        for (int i = 1; i < size - 2; ++i)
        {
            for (int k = i + 1; k < size - 1; ++k)
            {
                double delta = distance(route[i - 1], route[k], customers) + distance(route[i], route[k + 1], customers) -
                               (distance(route[i - 1], route[i], customers) + distance(route[k], route[k + 1], customers));
                if (delta < 0)
                {
                    two_opt_swap(route, i, k);
                    improvement = true;
                    iterations++;
                    if (iterations >= max_iterations)
                    {
                        break;
                    }
                }
            }
            if (iterations >= max_iterations)
            {
                break;
            }
        }
    }
}

// Function to perform del50: Remove worst 50% and replace with new unique individuals
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
    sort(indices.begin(), indices.end(), [&](int a, int b) -> bool
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

    // Create a list of cities (customers) from 2 to num_customers
    vector<int> remaining_cities;
    for (int i = 2; i <= num_customers; ++i)
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
        // Add depot (1) at the beginning and end
        route.insert(route.begin(), 1);
        route.push_back(1);

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

// Function to create a new population excluding specific sets (for complete reset)
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

    // Create a list of cities (customers) from 2 to num_customers
    vector<int> remaining_cities;
    for (int i = 2; i <= num_customers; ++i)
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
        // Add depot (1) at the beginning and end
        route.insert(route.begin(), 1);
        route.push_back(1);

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

// **GA algorithm to find the shortest route with Enhancements**
result solveGA(const ProblemData &data, int pop_size, int generations, double mutation_rate = 0.1)
{
    int num_customers = data.customers_count; // Now excludes separate depot
    const vector<Customers> &customers = data.customers;

    // Variables to track improvement
    double best_distance = numeric_limits<double>::max();
    vector<int> best_route;
    int generations_without_improvement = 0;
    const int max_generations_without_improvement = 50;

    try
    {
        // **Find Forbidden Transitions**
        unordered_map<int, unordered_set<int>> forbidden_transitions = find_long_distance_pairs(customers);
        cout << "Number of forbidden transitions: " << forbidden_transitions.size() << endl;

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
            throw std::runtime_error("Initial population creation failed");
        }

        // Initialize current mutation rate
        double current_mutation_rate = mutation_rate;

        // Initialize del50 counter
        int del50_counter = 0;

        // Initialize reset counter
        int reset_counter = 0;

        // Define elite size
        const int elite_size = 5;

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

            // Pair each individual with its fitness
            vector<pair<double, int>> fitness_pairs;
            fitness_pairs.reserve(population.size());
            for (int i = 0; i < fitnesses.size(); ++i)
            {
                fitness_pairs.emplace_back(fitnesses[i], i);
            }

            // Sort based on fitness in descending order
            sort(fitness_pairs.begin(), fitness_pairs.end(), [](const pair<double, int> &a, const pair<double, int> &b) -> bool
                 { return a.first > b.first; });

            // Extract elite individuals
            vector<vector<int>> elites;
            for (int i = 0; i < elite_size && i < fitness_pairs.size(); ++i)
            {
                elites.push_back(population[fitness_pairs[i].second]);
            }

            // **Apply 2-opt to Elites**
            for (auto &elite : elites)
            {
                two_opt(elite, customers, 1000);
            }

            // Find the best individual in the current generation
            double current_best_distance = numeric_limits<double>::max();
            int best_index = -1;

            for (int i = 0; i < fitness_pairs.size(); ++i)
            {
                int idx = fitness_pairs[i].second;
                double current_distance = total_distance(population[idx], customers);
                if (current_distance < current_best_distance)
                {
                    current_best_distance = current_distance;
                    best_index = idx;
                }
            }

            // Apply two_opt to the best individual
            if (best_index != -1)
            {
                two_opt(population[best_index], customers, 1000);
                // Recalculate the distance after two_opt
                current_best_distance = total_distance(population[best_index], customers);
            }

            // If a new best route was found, update best_route
            if (best_index != -1 && current_best_distance < best_distance)
            {
                best_distance = current_best_distance;
                best_route = population[best_index];
                generations_without_improvement = 0; // Reset the counter
            }
            else
            {
                generations_without_improvement++;
                // If no improvement after max_generations_without_improvement generations, apply del50
                if (generations_without_improvement >= max_generations_without_improvement)
                {
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

                        // Inform about the complete reset
                        cout << "Del50 has been invoked 5 times. Performing a complete population reset..." << endl;

                        // Generate a completely new population excluding initial and last populations
                        unordered_set<std::vector<int>, VectorHash> last_population_before_reset(population.begin(), population.end(), pop_size * 2, VectorHash());
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

            // **Genetic Operations: Selection, Crossover, Mutation with Elitism**

            vector<vector<int>> new_population_genetic;
            unordered_set<std::vector<int>, VectorHash> new_population_set_genetic;

            // Add elites to the new population
            for (const auto &elite : elites)
            {
                new_population_genetic.push_back(elite);
                new_population_set_genetic.insert(elite);
            }

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

                // **Selection: Tournament Selection**
                vector<int> parent1 = selection_tournament(population, fitnesses);
                vector<int> parent2 = selection_tournament(population, fitnesses);

                // **Crossover: Order Crossover (OX)**
                vector<int> child = crossover_order(parent1, parent2);

                // **Mutation: Apply Mutation Operators**
                mutate(child, current_mutation_rate, customers);

                // **Ensure Validity: No Forbidden Transitions, Uniqueness, and No -1 Values**
                if (!contains_forbidden_transition(child, forbidden_transitions) &&
                    find(child.begin(), child.end(), -1) == child.end())
                {
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
                population_set = std::move(new_population_set_genetic);
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

        // Ensure the route ends at depot (Customer 1)
        if (!best_route.empty() && best_route.back() != 1)
        {
            best_route.push_back(1);
            best_distance += distance(best_route[best_route.size() - 2], 1, customers);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nAn exception occurred during GA execution: " << e.what() << '\n';
        // Proceed to return the best result found so far
    }

    // Populate the result struct
    result result;
    result.path = best_route;
    result.lowest_distance = best_distance;

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

    // Run the GA algorithm
    int population_size = 100;   // Increased population size
    int generations = 1000;      // Increased number of generations
    double mutation_rate = 0.15; // Adjusted mutation rate

    result result; // Declare result here

    try
    {
        result = solveGA(data, population_size, generations, mutation_rate);

        // Print the results
        if (!result.path.empty())
        {
            cout << "\n\nBest distance: " << fixed << setprecision(2) << result.lowest_distance << endl;
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
                double d = distance(from, to, data.customers);
                cout << "Move from point " << from << " to point " << to << " | Distance: " << fixed << setprecision(2) << d << endl;
                total_distance_traveled += d;
            }
            cout << "Total Distance Traveled: " << fixed << setprecision(2) << total_distance_traveled << endl;
        }
        else
        {
            cout << "\nNo valid route found." << endl;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nAn exception occurred: " << e.what() << '\n';
        // Attempt to output the best result found so far
        if (!result.path.empty())
        {
            cout << "\n\nBest distance: " << fixed << setprecision(2) << result.lowest_distance << endl;
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
                double d = distance(from, to, data.customers);
                cout << "Move from point " << from << " to point " << to << " | Distance: " << fixed << setprecision(2) << d << endl;
                total_distance_traveled += d;
            }
            cout << "Total Distance Traveled: " << fixed << setprecision(2) << total_distance_traveled << endl;
        }
        else
        {
            cout << "\nNo valid route found." << endl;
        }
    }

    return 0;
}
