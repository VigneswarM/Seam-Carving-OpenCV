#include "sc.h"
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

/*** Vigneswar Mourouguessin  ***/

double Xval(Mat& iimage, int i, int j)
{
    Vec3b X1, X2;
    double xval = 0, gval = 0, bval = 0, rval = 0;

    if (i != 0)
        X2 = iimage.at<Vec3b>(i - 1, j);
    if (i != iimage.rows - 1)
        X1 = iimage.at<Vec3b>(i + 1, j);
    if (i == iimage.rows - 1)
        X1 = iimage.at<Vec3b>(0, j);
    if (i == 0)
        X2 = iimage.at<Vec3b>(iimage.rows - 1, j);

    rval = X1[2] - X2[2];
    gval = X1[1] - X2[1];
    bval = X1[0] - X2[0];
    xval = pow(gval, 2) + pow(bval, 2) + pow(rval, 2);
    return xval;
}

double Yval(Mat& iimage, int i, int j)
{
    Vec3b Y1, Y2;
    double yval = 0, gval = 0, bval = 0, rval = 0;

    if (j != iimage.cols - 1)
        Y1 = iimage.at<Vec3b>(i, j + 1);
    if (j != 0)
        Y2 = iimage.at<Vec3b>(i, j - 1);
    if (j == 0)
        Y2 = iimage.at<Vec3b>(i, iimage.cols - 1);
    if (j == iimage.cols - 1)
        Y1 = iimage.at<Vec3b>(i, 0);

    rval = Y1[2] - Y2[2];
    gval = Y1[1] - Y2[1];
    bval = Y1[0] - Y2[0];
    yval = pow(gval, 2) + pow(bval, 2) + pow(rval, 2);
    return yval;
}

double Compute_val(Mat& iimage, int i, int j)
{
    double xGrad = Xval(iimage, i, j);
    double yGrad = Yval(iimage, i, j);
    return xGrad + yGrad;
}

void Compute_Seam(int Threshold, vector<vector<int> >& Energy, vector<int>& Seam, int rows, int cols)
{
    int SeamLine = 0, val = rows - 1, flag = 0, index = val - 1;
    int visited = 0, count = 0, threshold = cols - 1;
    bool check = false;

    for (int in = 1; in < cols; in++) {
        int prevThres = Threshold;
        Threshold = min(prevThres, Energy[val][in]);

        if (in == threshold)
            check = true;

        if (prevThres != Threshold) {
            SeamLine = in;
            Seam[index] = in;
            count++;
        }
    }

    while (flag != 1) {
        if (check) {
            if (index > 0) {

                int line1 = SeamLine - 1;
                int line2 = SeamLine + 1;

                if (SeamLine == 0 && visited == 0) {
                    if (Energy[index][SeamLine] > Energy[index][line2] && visited == 0)
                        SeamLine++;
                    visited = 1;
                }
                else if (SeamLine == val + 1 && visited == 0) {
                    if (Energy[index][line1] < Energy[index][SeamLine] && visited == 0)
                        SeamLine--;
                    visited = 1;
                }
                else if (Energy[index][line1] < Energy[index][SeamLine] && visited == 0) {
                    if (Energy[index][line1] < Energy[index][line2] && visited == 0) {
                        SeamLine--;
                        visited = 1;
                    }
                }
                else if (Energy[index][line1] < Energy[index][SeamLine] && visited == 0) {
                    if (Energy[index][line1] > Energy[index][line2] && visited == 0) {
                        SeamLine++;
                        visited = 1;
                    }
                }
                else {
                    if (Energy[index][SeamLine] > Energy[index][line2] && visited == 0)
                        SeamLine++;
                    visited = 1;
                }
                Seam[index] = SeamLine;
                index--;
                visited = 0;
            }
            else
                flag = 1;
        }
    }
}

int Compute_Energies(int rows, int cols, vector<vector<int> >& Energy, Mat& iimage)
{
    int count = rows * cols;
    vector<int> Energies;
    int minW = 999999999;
    int flag = 0, check = 0;
    int Sem;

    for (int id = 0; id < count; id++) {
        int i = id / cols;
        int j = id % cols;
        double Threshold = Compute_val(iimage, i, j);

        if (i == 0) {
            Energy[i][j] = Threshold;
            minW = Threshold;
            check = 1;
        }
        else {
            if (j != cols - 1 && j != 0 && check == 1) {
                int temp = min(Energy[i - 1][j - 1], Energy[i - 1][j]);
                int prev = min(temp, Energy[i - 1][j + 1]);
                Energy[i][j] = Threshold + prev;
                Energies.push_back(Threshold + prev);
            }
            if (j == 0 && check == 1) {
                int prev = min(Energy[i - 1][j], Energy[i - 1][j + 1]);
                Energies.push_back(Threshold + prev);
                Energy[i][j] = Threshold + prev;
            }
            if (i == 0 && check == 1) {
                int prev = 0;
                Energies.push_back(Threshold + prev);
                Energy[i][j] = Threshold + prev;
            }
            if (j == cols - 1 && check == 1) {
                int prev = min(Energy[i - 1][j - 1], Energy[i - 1][j]);
                Energies.push_back(Threshold + prev);
                Energy[i][j] = Threshold + prev;
            }
            if (i == rows - 1 && j == 0) {
                minW = Energy[i][j];
                flag = 1;
            }
            if (j > 0 && flag == 1) {
                if (i == rows - 1 && j == cols - 1) {
                    minW = Energy[i][j];
                    Sem = j;
                }
                else if (minW > Energy[i][j]) {
                    minW = Energy[i][j];
                    Sem = j;
                }
            }
        }
    }
    return minW;
}

bool reduce_seam(Mat& iimage, Mat& oimage, int rows, int cols)
{

    int visited = 0, k, i = rows - 1, j, l = 0;
    rows = iimage.rows;
    vector<int> Seam(rows);
    cols = iimage.cols;
    oimage.create(rows, cols - 1, CV_8UC3);
    Mat Reduced_image = Mat(rows, cols - 1, CV_8UC3);
    vector<vector<int> > Energy(rows, vector<int>(cols));
    int count = rows * cols;
    int val = cols - 1;
    int Threshold = Compute_Energies(rows, cols, Energy, iimage);
    Compute_Seam(Threshold, Energy, Seam, rows, cols);
    bool out_f = true;

    while (i >= 0) {
        for (j = 0, k = 0; j < cols; j++) {
            if (j != Seam[i] && k < val && i >= 0) {
                Reduced_image.at<Vec3b>(i, k) = iimage.at<Vec3b>(i, j);
                k++;
            }
            else {
                oimage.at<Vec3b>(i, j) = iimage.at<Vec3b>(i, j);
                l++;
            }
        }
        --i;
    }
    Reduced_image.copyTo(oimage);
    Seam.clear();
    Energy.clear();
    return out_f;
}

void compute_VSeam(Mat& iimage, Mat& oimage, int newWidth, int VSeam)
{
    int rows = iimage.rows;
    int cols = iimage.cols;
    while (iimage.cols != newWidth) {
        if (iimage.cols > newWidth) {
            VSeam++;
            reduce_seam(iimage, oimage, rows, cols);
            iimage = oimage.clone();
            cout << "Width:" << oimage.cols << endl;
        }
    }
}

void compute_HSeam(Mat& iimage, Mat& oimage, int newHeight, int HSeam)
{
    int rows = iimage.rows;
    int cols = iimage.cols;
    while (iimage.cols != newHeight) {
        if (iimage.cols > newHeight) {
            HSeam++;
            reduce_seam(iimage, oimage, rows, cols);
            iimage = oimage.clone();
            cout << "Height:" << oimage.cols << endl;
        }
    }
}

bool seam_carving_trivial(Mat& inImage, Mat& outImage, int newWidth, int newHeight)
{
    Mat input = inImage.clone();
    Mat output = inImage.clone();
    bool out_f = true;
    int VSeam = 0;
    compute_VSeam(input, output, newWidth, VSeam);
    rotate(input, output, ROTATE_90_CLOCKWISE);
    output.copyTo(input);
    int HSeam = 0;
    compute_HSeam(input, output, newHeight, HSeam);
    rotate(input, output, ROTATE_90_COUNTERCLOCKWISE);
    output.copyTo(input);
    outImage = output.clone();
    return out_f;
}

bool seam_carving(Mat& input_Image, int newWidth, int newHeight, Mat& output_Image)
{
    int cols = input_Image.cols;
    int rows = input_Image.rows;

    if (newHeight <= 0) {
        cout << "Invalid request!!! newHeight has to be positive!" << endl;
        return false;
    }

    if (newWidth > cols) {
        cout << "Invalid request!!! new Width has to be smaller than the current size!" << endl;
        return false;
    }

    if (newWidth <= 0) {
        cout << "Invalid request!!! newWidth has to be positive!" << endl;
        return false;
    }

    if (newHeight > rows) {
        cout << "Invalid request!!! new Height has to be smaller than the current size!" << endl;
        return false;
    }
    return seam_carving_trivial(input_Image, output_Image, newWidth, newHeight);
}
