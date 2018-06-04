/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <random> // Need this for sampling from distributions
#include <cfloat>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 100;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	particles.resize( num_particles );
	weights.resize(   num_particles, 1.0 );

	for(int i=0; i<num_particles; i++){
		particles[i].id	    = i;
		particles[i].weight = 1.0;
	        particles[i].x      = dist_x(gen);
	        particles[i].y      = dist_y(gen);
	        particles[i].theta  = dist_theta(gen);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;

	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	for(int i=0; i<num_particles; i++){
		particles[i].x     += velocity / yaw_rate * ( sin( particles[i].theta + yaw_rate * delta_t ) - sin( particles[i].theta ) ) + dist_x(gen);
		particles[i].y     += velocity / yaw_rate * ( cos( particles[i].theta ) - cos( particles[i].theta + yaw_rate * delta_t ) ) + dist_y(gen);
		particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.




}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double gauss_norm = 0.5 / ( M_PI * std_landmark[0] * std_landmark[1] );	//calculate normalization term

	for(int i=0; i<num_particles; i++){

		double weight = 1.0;

		for(int j=0; j<observations.size(); j++){

			//transformation from vehicle's coordinate to map's coordinate
			double sint = sin( particles[i].theta );
			double cost = cos( particles[i].theta );

			double xmap = particles[i].x + cost * observations[j].x - sint * observations[j].y;
			double ymap = particles[i].y + sint * observations[j].x + cost * observations[j].y;


			//Associate the nearest landmark
			double dist_min = DBL_MAX;
			double xland = 0.0;
			double yland = 0.0;
			for(int k=0; k<map_landmarks.landmark_list.size(); k++){
				double distance = dist(map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f, xmap, ymap);
				if( distance < dist_min ){
					dist_min = distance;
					xland    = map_landmarks.landmark_list[k].x_f;
					yland    = map_landmarks.landmark_list[k].y_f;
				}
			}

			//Calculate Weight
			double diff_x = xmap - xland;
			double diff_y = ymap - yland;

			double exponent_x = diff_x * diff_x / (2 * std_landmark[0] * std_landmark[0]);
			double exponent_y = diff_y * diff_y / (2 * std_landmark[1] * std_landmark[1]);
			double exponent   = exponent_x + exponent_y;

			weight *= gauss_norm * exp( -exponent );
		}

		particles[i].weight = weight;
		weights[i]          = weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	random_device seed_gen;
	mt19937 engine( seed_gen() );
	std::vector<Particle> new_particles( num_particles );

	discrete_distribution<std::size_t> distrib( weights.begin(), weights.end() );

	for(int i=0; i<num_particles; i++ ){
		size_t index = distrib( engine );
		new_particles[i] = particles[index];
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
