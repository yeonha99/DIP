#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include<opencv.hpp>
#include <math.h>
using namespace cv;

using std::cout;
using std::endl;

bool R1(int R, int G, int B) {
	bool e1 = (R > 95) && (G > 40) && (B > 20) && ((max(R, max(G, B)) - min(R, min(G, B))) > 20) && (abs(R - G) > 20) && (R > G) && (R > B);
	bool e2 = (R > 220) && (G > 210) && (B > 170) && (abs(R - G) <= 15) && (R > B) && (G > B);
	return (e1 || e2);
}

bool R2(float Y, float Cr, float Cb) {
	bool e3 = Cr <= 1.5862 * Cb + 55;
	bool e4 = Cr >= 0.3448 * Cb + 86.2069;
	bool e5 = Cr >= -4.5652 * Cb + 334.5652;
	bool e6 = Cr <= -1.15 * Cb + 301.75;
	bool e7 = Cr <= -2.2857 * Cb + 432.85;
	return e3 && e4 && e5 && e6 && e7;
}

bool R3(float H, float S, float V) {
	return (H < 25) || (H > 330);
}

Mat RGBcolor_space(Mat const& src) {
	// allocate the result matrix
	Mat dst = src.clone();

	Vec3b cwhite = Vec3b::all(255);
	Vec3b cblack = Vec3b::all(0);
	Mat src_ycrcb, src_hsv;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
			int B = pix_bgr.val[0];
			int G = pix_bgr.val[1];
			int R = pix_bgr.val[2];
			// apply rgb rule
			bool a = R1(R, G, B);

			if (!a)
				dst.ptr<Vec3b>(i)[j] = cblack;
		}
	}
	return dst;
}


Mat HSI_color_space(Mat const& src) {
	// allocate the result matrix
	Mat dst = src.clone();

	Vec3b cwhite = Vec3b::all(255);
	Vec3b cblack = Vec3b::all(0);
	Mat src_hsv;
	src.convertTo(src_hsv, CV_32FC3);
	cvtColor(src_hsv, src_hsv, COLOR_BGR2HSV);
	normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
			float H = pix_hsv.val[0];
			float S = pix_hsv.val[1];
			float V = pix_hsv.val[2];
			// apply hsv rule
			bool c = R3(H, S, V);

			if (c)
				dst.ptr<Vec3b>(i)[j] = cblack;
		}
	}
	return dst;
}

Mat YCrCb_color_space(Mat const& src) {
	// allocate the result matrix
	Mat dst = src.clone();

	Vec3b cwhite = Vec3b::all(255);
	Vec3b cblack = Vec3b::all(0);

	Mat src_ycrcb, src_hsv;
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
			int Y = pix_ycrcb.val[0];
			int Cr = pix_ycrcb.val[1];
			int Cb = pix_ycrcb.val[2];
			bool b = R2(Y, Cr, Cb);
			if (b) {
				dst.ptr<Vec3b>(i)[j] = cblack;
			}

		}
	}
	return dst;
}
Mat GetSkin(Mat const& src) {
	// allocate the result matrix
	Mat dst = src.clone();
	Vec3b cwhite = Vec3b::all(255);
	Vec3b cblack = Vec3b::all(0);
	Mat src_ycrcb, src_hsv;
	cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);
	src.convertTo(src_hsv, CV_32FC3);
	cvtColor(src_hsv, src_hsv, COLOR_BGR2HSV);
	normalize(src_hsv, src_hsv, 0.0, 255.0, NORM_MINMAX, CV_32FC3);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {

			Vec3b pix_bgr = src.ptr<Vec3b>(i)[j];
			int B = pix_bgr.val[0];
			int G = pix_bgr.val[1];
			int R = pix_bgr.val[2];
			// apply rgb rule
			bool a = R1(R, G, B);
			Vec3b pix_ycrcb = src_ycrcb.ptr<Vec3b>(i)[j];
			int Y = pix_ycrcb.val[0];
			int Cr = pix_ycrcb.val[1];
			int Cb = pix_ycrcb.val[2];
			// apply ycrcb rule
			bool b = R2(Y, Cr, Cb);

			Vec3f pix_hsv = src_hsv.ptr<Vec3f>(i)[j];
			float H = pix_hsv.val[0];
			float S = pix_hsv.val[1];
			float V = pix_hsv.val[2];
			// apply hsv rule
			bool c = R3(H, S, V);

			if (!(a && b && c))
				dst.ptr<Vec3b>(i)[j] = cblack;
		}
	}
	return dst;
}
int main() {
	// Get filename to the source image:
	Mat image = imread("C:\\Users\\JungYeonHee\\source\\repos\\DIP_project1\\DIP_project1\\deformation_7.jpg", 1);
	Mat src;
	Mat skin;
	Mat dst1;
	Mat dst1_1;
	Mat dst2;
	Mat dst2_1;
	Mat dst3;
	Mat dst3_1;
	Mat gray;
	Mat base;
	cv::GaussianBlur(image, image, cv::Size(5, 5), 1.5);
	cv::resize(image, src, cv::Size(image.cols *1, image.rows *1), 0, 0);
	skin = GetSkin(src);
	dst1 = RGBcolor_space(src);
	//threshold(dst1, dst1_1, 0, 255, THRESH_BINARY);
	//imshow("RGB_mediaBlur before", dst1_1);//blur하기 전
	cvtColor(dst1, dst1, COLOR_BGR2GRAY);
	dst2 = HSI_color_space(src);
	threshold(dst1, dst1, 112, 255, THRESH_BINARY);
	cv::medianBlur(dst1, dst1, 5);

	cvtColor(dst2, dst2, COLOR_BGR2GRAY);
	//threshold(dst2, dst2_1, 0, 255, THRESH_BINARY);
	//imshow("HSI_mediaBlur before", dst2_1);//blur하기 전
	cv::medianBlur(dst2, dst2, 5);
	threshold(dst2, dst2, 0, 255, THRESH_BINARY);

	//dst3 = YCrCb_color_space(src);
	//cv::medianBlur(dst3, dst3, 5);

	dst3 = YCrCb_color_space(src);
	//threshold(dst3, dst3_1, 0, 255, THRESH_BINARY);
	//imshow("YCrCb_mediaBlur before", dst3_1);//blur하기 전
	cvtColor(dst3, dst3, COLOR_BGR2GRAY);
	cv::medianBlur(dst3, dst3, 5);
	threshold(dst3, dst3, 0, 255, THRESH_BINARY);


	namedWindow("original");
	namedWindow("RGBcolor_space");
	namedWindow("HSI_color_space");
	namedWindow("YCrCb_color_space");
	namedWindow("raw detected sample");
	imshow("original", src);
	imshow("RGBcolor_space", dst1);
	imshow("HSI_color_space", dst2);
	imshow("YCrCb_color_space", dst3);
	imshow("raw detected sample", skin);
	waitKey(0);
	return 0;
}