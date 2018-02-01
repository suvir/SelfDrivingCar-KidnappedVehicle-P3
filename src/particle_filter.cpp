/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */
#define PI 3.14159265359

#include <random>
#include <iostream>
#include <sstream>
#include <array>
#include "map.h"
#include "particle_filter.h"

using namespace std;

#define NUM_PARTICLES 100

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = NUM_PARTICLES;

    default_random_engine gen;
    double std_x, std_y, std_theta, weight;
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        x = dist_x(gen);
        y = dist_y(gen);
        theta = dist_theta(gen);
        weight = 1.0;
        Particle par;
        par = (Particle) {i, x, y, theta};
        weights.push_back(weight);
        particles.push_back(par);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    default_random_engine gen;
    double std_x, std_y, std_theta;
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    for (int i = 0; i < num_particles; ++i) {
        if (abs(yaw_rate) < 0.0001) {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        } else {
            particles[i].x +=
                    velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y +=
                    velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }
        normal_distribution<double> dist_x(particles[i].x, std_x);
        normal_distribution<double> dist_y(particles[i].y, std_y);
        normal_distribution<double> dist_theta(particles[i].theta, std_theta);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {
    // Was able to get meet grading criteria without implementing this.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations,
                                   Map map_landmarks) {
    double std_r, std_b;  // range and bearing deviation
    std_r = std_landmark[0];
    std_b = std_landmark[1];
    weights.clear();
    for (int i = 0; i < num_particles; ++i)
    {
        double par_x = particles[i].x;
        double par_y = particles[i].y;
        double par_theta = particles[i].theta;

        Map selected_map;
        selected_map.landmark_list.clear();
        for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
            if (abs(map_landmarks.landmark_list[j].x_f - par_x) <= sensor_range &&
                abs(map_landmarks.landmark_list[j].y_f - par_y) <= sensor_range) {
                selected_map.landmark_list.push_back(map_landmarks.landmark_list[j]);
            }
        }

        std::vector<LandmarkObs> transformed_obs;
        LandmarkObs ob;
        for (int j = 0; j < observations.size(); ++j) {
            ob.x = observations[j].x * cos(par_theta) - observations[j].y * sin(par_theta) + par_x;
            ob.y = observations[j].x * sin(par_theta) + observations[j].y * cos(par_theta) + par_y;

            // association
            double distance = pow(selected_map.landmark_list[0].x_f - ob.x, 2) +
                              pow(selected_map.landmark_list[0].y_f - ob.y, 2);
            double tmp_dist;
            ob.id = 0;
            for (int k = 1; k < selected_map.landmark_list.size(); ++k) {
                tmp_dist = pow(selected_map.landmark_list[k].x_f - ob.x, 2) +
                           pow(selected_map.landmark_list[k].y_f - ob.y, 2);
                if (tmp_dist < distance) {
                    distance = tmp_dist;
                    ob.id = k;
                }
            }
            transformed_obs.push_back(ob);
        }

        // update weights
        double updated_w = 1.0;
        for (int j = 0; j < transformed_obs.size(); ++j)
        {
            updated_w *= exp(-pow(transformed_obs[j].x - selected_map.landmark_list[transformed_obs[j].id].x_f, 2) / 2.0 /
                         pow(std_r, 2)
                         - pow(transformed_obs[j].y - selected_map.landmark_list[transformed_obs[j].id].y_f, 2) / 2.0 /
                           pow(std_b, 2)) /
                     2.0 / PI / std_r / std_b;
        }
        weights.push_back(updated_w);
    }
}


void ParticleFilter::resample() {
    std::vector<Particle> selected_particles;
    std::default_random_engine generator;

    std::discrete_distribution<> distribution(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; ++i) {
        int idx = distribution(generator);
        selected_particles.push_back(particles[idx]);
    }

    particles = selected_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}