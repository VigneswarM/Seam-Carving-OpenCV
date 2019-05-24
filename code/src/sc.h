#ifndef SEAMCARVINGCOMP665156
#define SEAMCARVINGCOMP665156
#include <opencv2/opencv.hpp>

/*** Vigneswar Mourouguessin ***/

bool seam_carving(cv::Mat& in_image, int new_width, int new_height,cv::Mat& out_image);
bool seam_carving_trivial(cv::Mat& inImage,cv::Mat& outImage, int newWidth, int newHeight);

double Xval(cv::Mat& iimage, int i, int j);
double Yval(cv::Mat& iimage, int i, int j);
double Compute_val(cv::Mat& iimage, int i, int j);

#endif
