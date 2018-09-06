#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#include <flow2pose_lib/flow2pose.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Pose3.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace gtsam;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

string parse_argument(int argc, char* argv[]) {
    po::options_description desc("Allowed options");

    string correspondence_path;

    desc.add_options()
        ("help", "produce help message")
        ("c", po::value<string>(&correspondence_path)->default_value(""),
         "correspondence fields");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    return correspondence_path;
}

void parse_file(const string& file, vector<Point3>& pts1, vector<Point3>& pts2) {

  cout << "parse file" << file << endl;

  ifstream infile(file);
  string line;
  while( getline(infile, line) ) {
    double x0, y0, z0, x1, y1, z1;
    istringstream(line) >> x0 >> y0 >> z0 >> x1 >> y1 >> z1;

    pts1.push_back(Point3(x0, y0, z0));
    pts2.push_back(Point3(x1, y1, z1));
  }
}

int main(int argc, char* argv[]) {
    auto string = parse_argument(argc, argv);

    vector<Point3> pts1;
    vector<Point3> pts2;
    parse_file(string, pts1, pts2);

    cout << "estimate pose with " << pts1.size() << " points pairs" << endl;

    Flow2Pose flow2pose;
    Pose3 raw_pose = Pose3::Create(Rot3::Rodrigues(0,0,0), Point3(0, 0, 0));
    Pose3 est_pose = flow2pose.solve(pts1, pts2, raw_pose);

    est_pose.print("estimated pose matrix is: ");
}
