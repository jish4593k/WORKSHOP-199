#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

unordered_map<string, string> var_dict;

string process_variables(string arg) {
    for (const auto& [key, value] : var_dict) {
        size_t found = arg.find("{" + key + "}");
        while (found != string::npos) {
            arg.replace(found, key.length() + 3, value);
            found = arg.find("{" + key + "}");
        }
    }
    return arg;
}

void train_regression_model(MatrixXd& X, VectorXd& y, NeuralNetwork& model) {
    model.addLayer(new DenseLayer(64, "relu", X.cols()));
    model.addLayer(new DenseLayer(1, "", 64));

    model.compile("adam", "mean_squared_error");
    model.fit(X, y, 50, 32, 0.2);
}

int main() {
    vector<vector<string>> data;

    ifstream file("main.csv");
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<string> tokens{istream_iterator<string>{iss}, istream_iterator<string>{}};
        data.push_back(tokens);
    }

    NeuralNetwork regression_model;

    for (const auto& row : data) {
        string command = row[0];
        vector<string> args(row.begin() + 1, row.end());

        if (command == "let") {
            var_dict[args[0]] = args[1];
        }

        if (command == "print") {
            for (const auto& arg : args) {
                if (arg.find("{") != string::npos && arg.find("}") != string::npos) {
                    cout << process_variables(arg) << " ";
                } else {
                    cout << arg << " ";
                }
            }
            cout << endl;
        }
    }

    // Dummy regression data, replace with your own dataset
    MatrixXd X_regression = MatrixXd::Random(100, 1);
    VectorXd y_regression = 3 * X_regression.col(0).array() + 2 + 0.1 * VectorXd::Random(100);

    // Train a regression model
    train_regression_model(X_regression, y_regression, regression_model);

    // Example: Predict using the trained regression model
    MatrixXd features(1, 1);
    features << 0.5;
    MatrixXd prediction = regression_model.predict(features);
    cout << "Regression Prediction: " << prediction(0, 0) << endl;

    return 0;
}
