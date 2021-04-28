#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <iostream>
#include <vector>
#include <fstream>
#include <queue>
using namespace std;
using namespace cv;
bool visit[400][400];
Mat segmentation(Mat& src, Mat& dst) {

	src.convertTo(dst, CV_8UC1);
	pyrUp(dst, dst, Size(1 / 4, 1 / 4));
	medianBlur(dst, dst, 7);

	Mat mask = getStructuringElement(1, Size(3, 3), Point(1, 1));
	adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 2);

	Mat seg;
	morphologyEx(dst, seg, MORPH_OPEN, mask, Point(-1, -1), 12);

	GaussianBlur(dst, dst, Size(9, 9), 7);
	adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 1);

	erode(dst, dst, mask, Point(-1, -1), 1);
	pyrDown(dst, dst, Size(1 / 4, 1 / 4));
	pyrDown(seg, seg, Size(1 / 4, 1 / 4));
	dst = seg;
	threshold(seg, seg, 127, 255, THRESH_BINARY_INV);
	return seg;
}
pair<Mat, vector<pair<float, float>>> orientation(Mat src, int size, int* SP, int X[], int Y[], unsigned char O[], unsigned char T[], int& core_check, int& delta_check) {
	Mat inputImage = src;

	inputImage.convertTo(inputImage, CV_32F, 1.0 / 255, 0);

	medianBlur(inputImage, inputImage, 3);

	int blockSize = size;

	Mat fprintWithDirectionsSmoo = inputImage.clone();
	cvtColor(fprintWithDirectionsSmoo, fprintWithDirectionsSmoo, COLOR_GRAY2BGR);

	Mat tmp(inputImage.size(), inputImage.type());
	Mat coherence(inputImage.size(), inputImage.type());
	Mat orientationMap = tmp.clone();

	//Gradiants x and y
	Mat grad_x, grad_y;
	Sobel(inputImage, grad_x, inputImage.depth(), 1, 0, 3);
	Sobel(inputImage, grad_y, inputImage.depth(), 0, 1, 3);
	Mat Fx(inputImage.size(), inputImage.type()),
		Fy(inputImage.size(), inputImage.type()),
		Fx_gauss,
		Fy_gauss;
	Mat copy_input(inputImage.size(), inputImage.type());

	int width = inputImage.cols;
	int height = inputImage.rows;
	int blockH;
	int blockW;

	vector<pair<float, float>> vec;
	vector<int> cnt;
	for (int i = 0; i < height; i += blockSize) {
		for (int j = 0; j < width; j += blockSize) {
			float Gsx = 0.0;
			float Gsy = 0.0;
			float Gxx = 0.0;
			float Gyy = 0.0;

			//for check bounds of img
			blockH = ((height - i) < blockSize) ? (height - i) : blockSize;
			blockW = ((width - j) < blockSize) ? (width - j) : blockSize;

			//average at block WхW
			for (int u = i; u < i + blockH; u++) {
				for (int v = j; v < j + blockW; v++) {
					Gsx += (grad_x.at<float>(u, v) * grad_x.at<float>(u, v)) - (grad_y.at<float>(u, v) * grad_y.at<float>(u, v));
					Gsy += 2 * grad_x.at<float>(u, v) * grad_y.at<float>(u, v);
					Gxx += grad_x.at<float>(u, v) * grad_x.at<float>(u, v);
					Gyy += grad_y.at<float>(u, v) * grad_y.at<float>(u, v);
				}
			}

			float coh = sqrt(pow(Gsx, 2) + pow(Gsy, 2)) / (Gxx + Gyy);
			//copy_input
			float fi = 0.5f * fastAtan2(Gsy, Gsx) * CV_PI / 180.0f;

			Fx.at<float>(i, j) = cos(2 * fi);
			Fy.at<float>(i, j) = sin(2 * fi);

			//fill blocks
			for (int u = i; u < i + blockH; u++) {
				for (int v = j; v < j + blockW; v++) {
					orientationMap.at<float>(u, v) = fi;
					Fx.at<float>(u, v) = Fx.at<float>(i, j);
					Fy.at<float>(u, v) = Fy.at<float>(i, j);
					coherence.at<float>(u, v) = (coh < 0.85f) ? 1.0f : 0.0f;
				}
			}
		}
	}

	GaussianBlur(Fx, Fx_gauss, Size(5, 5), 1, 1);
	GaussianBlur(Fy, Fy_gauss, Size(5, 5), 1, 1);
	vector<vector<float>> vec_gradgrad(height, vector<float>(width, 0.0f));
	for (int m = 0; m < height; m++) {
		for (int n = 0; n < width; n++) {
			copy_input.at<float>(m, n) = 0.5f * fastAtan2(Fy_gauss.at<float>(m, n), Fx_gauss.at<float>(m, n)) * CV_PI / 180.0f;
			if ((m % blockSize) == 0 && (n % blockSize) == 0) {
				int x = n;
				int y = m;
				int ln = sqrt(2 * pow(blockSize, 2)) / 2;
				float dx = ln * cos(copy_input.at<float>(m, n) - CV_PI / 2.0f);
				float dy = ln * sin(copy_input.at<float>(m, n) - CV_PI / 2.0f);
				vec.push_back({ dx,dy });

				float grad = dy / (dx + FLT_EPSILON);

				float gradgrad = grad;

				// 4개로 qusntazation
				if (2.0f <= gradgrad)
					gradgrad = FLT_MAX;
				else if (0.5f <= gradgrad && gradgrad < 2.0f)
					gradgrad = 1.0f;
				else if (-0.5f <= gradgrad && gradgrad < 0.5f)
					gradgrad = 0.0f;
				else if (-2.0f <= gradgrad && gradgrad < -0.5f)
					gradgrad = -1.0f;
				else if (gradgrad < -2.0f)
					gradgrad = FLT_MAX;

				vec_gradgrad[m][n] = gradgrad;

				int xx = (blockH / 2) / sqrt(pow(grad, 2) + 1);
				int yy = grad * xx;
				int mid_x = n + blockH / 2;
				int mid_y = m + blockH / 2;
				if (xx == 0 && yy == 0)
					yy = blockH / 2;

				line(fprintWithDirectionsSmoo, Point(mid_x + xx, mid_y + yy), Point(mid_x - xx, mid_y - yy), Scalar(0, 0, 255), 1, LINE_AA, 0);

			}
		}
	}

	normalize(orientationMap, orientationMap, 0, 1, NORM_MINMAX);

	orientationMap = copy_input.clone();

	normalize(copy_input, copy_input, 0, 1, NORM_MINMAX);

	pyrUp(copy_input, copy_input);

	pair<Mat, vector<pair<float, float>>> returning;

	returning = { fprintWithDirectionsSmoo, vec };

	return returning;
}
Mat gabor(Mat src, vector<pair<float, float>>& vec, int block_size) {
	Mat dst = Mat::zeros(src.rows, src.cols, CV_32F);

	int size = 15;
	double sigma = 5;
	double theta = 0;
	double lambd = 7;
	double gamma = 1;
	double psi = 0;

	int height = src.rows;
	int width = src.cols;
	int index = 0;

	for (int m = 0; m < height; m = block_size + m) {
		for (int n = 0; n < width; n = block_size + n) {
			/*      if (index >= 100)break;
				 cout << index << " ";*/
			float dx = vec[index].first;
			float dy = vec[index].second;

			// 해당 Block의 방향대로 theta를 설정해줌
			theta = atan2f(dy, dx) + CV_PI / 2;

			Mat temp;
			Mat gabor = getGaborKernel({ size, size }, sigma, theta, lambd, gamma, psi);
			filter2D(src, temp, CV_32F, gabor);

			int temp_size = block_size - 1;
			if (width < n + temp_size)
				temp_size = (width - 1) - n;
			if (height < m + block_size - 1 && temp_size >(height - 1) - m)
				temp_size = (height - 1) - m;

			// 해당 block만 이미지를 저장
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (m <= i && i <= m + temp_size && n <= j && j <= n + temp_size)
						dst.at<float>(i, j) = temp.at<float>(i, j);
				}
			}

			index++;

		}
	}

	dst.convertTo(dst, CV_8U);
	adaptiveThreshold(dst, dst, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);

	return dst;
}
void thinningIteration(cv::Mat& im, int iter)
{
	cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

	for (int i = 1; i < im.rows - 1; i++)
	{
		for (int j = 1; j < im.cols - 1; j++)
		{
			uchar p2 = im.at<uchar>(i - 1, j);
			uchar p3 = im.at<uchar>(i - 1, j + 1);
			uchar p4 = im.at<uchar>(i, j + 1);
			uchar p5 = im.at<uchar>(i + 1, j + 1);
			uchar p6 = im.at<uchar>(i + 1, j);
			uchar p7 = im.at<uchar>(i + 1, j - 1);
			uchar p8 = im.at<uchar>(i, j - 1);
			uchar p9 = im.at<uchar>(i - 1, j - 1);

			int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
				(p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
				(p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
				(p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
			int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
			int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
			int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0) {
				marker.at<uchar>(i, j) = 1;
			}
		}
	}

	im &= ~marker;
}

Mat thinning(const cv::Mat& src)
{
	Mat dst;
	dst = src.clone();
	dst /= 255;         // convert to binary image

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;

	do {
		thinningIteration(dst, 0);
		thinningIteration(dst, 1);
		cv::absdiff(dst, prev, diff);
		dst.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);


	dst *= 255;

	return dst;
}

void find_other_maniture(Mat& color, Mat& imgt, int& endingcheck, int& bifcheck, int& SP, int X[], int Y[], unsigned char O[], unsigned char T[], vector<pair<float, float>>& vec) {
	int a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0;
	int ending = 0;
	int bif = 0;
	long long ind = 0;

	for (int y = 1; y < imgt.rows - 1; y = y + 1)
	{
		for (int x = 1; x < imgt.cols - 1; x = x + 1)
		{

			if ((x % 14) == 0 && (y % 14) == 0)ind++;///blocksize*2기준으로 기울기 index조정 지금 이미지 사이즈가 2배라서 그럼
			if (SP > 49)break;
			if (ind >= vec.size())break;
			int save1 = imgt.at<uchar>(y - 1, x - 1);
			int save2 = imgt.at<uchar>(y - 1, x);
			int save3 = imgt.at<uchar>(y - 1, x + 1);
			int save4 = imgt.at<uchar>(y, x - 1);
			int save5 = imgt.at<uchar>(y, x);
			int save6 = imgt.at<uchar>(y, x + 1);
			int save7 = imgt.at<uchar>(y + 1, x - 1);
			int save8 = imgt.at<uchar>(y + 1, x);
			int save9 = imgt.at<uchar>(y + 1, x + 1);

			if (save1 > 0) a = 1;
			if (save2 > 0) b = 1;
			if (save3 > 0) c = 1;
			if (save4 > 0) d = 1;
			if (save5 > 0) e = 1;
			if (save6 > 0) f = 1;
			if (save7 > 0) g = 1;
			if (save8 > 0) h = 1;
			if (save9 > 0) i = 1;

			if (visit[y][x] == 0) {
				if (a == 1 && b == 0 && c == 0 && d == 0 && e == 1 && f == 0 && g == 0 && h == 0 && i == 0) {
					ending++;
				}
				if (a == 0 && b == 1 && c == 0 && d == 0 && e == 1 && f == 0 && g == 0 && h == 0 && i == 0) {
					ending++;
				}
				if (a == 0 && b == 0 && c == 1 && d == 0 && e == 1 && f == 0 && g == 0 && h == 0 && i == 0) {
					ending++;
				}
				if (a == 0 && b == 0 && c == 0 && d == 1 && e == 1 && f == 0 && g == 0 && h == 0 && i == 0) {
					ending++;
				}
				if (a == 0 && b == 0 && c == 0 && d == 0 && e == 1 && f == 1 && g == 0 && h == 0 && i == 0) {
					ending++;
				}
				if (a == 0 && b == 0 && c == 0 && d == 0 && e == 1 && f == 0 && g == 1 && h == 0 && i == 0) {
					ending++;
				}
				if (a == 0 && b == 0 && c == 0 && d == 0 && e == 1 && f == 0 && g == 0 && h == 1 && i == 0) {
					ending++;
				}
				if (a == 0 && b == 0 && c == 0 && d == 0 && e == 1 && f == 0 && g == 0 && h == 0 && i == 1) {
					ending++;
				}
			}


			if (a == 1 && b == 0 && c == 0 && d == 0 && e == 1 && f == 1 && g == 1 && h == 0 && i == 0) bif++;
			if (a == 0 && b == 1 && c == 0 && d == 1 && e == 1 && f == 0 && g == 0 && h == 0 && i == 1) bif++;
			if (a == 1 && b == 0 && c == 1 && d == 0 && e == 1 && f == 0 && g == 0 && h == 1 && i == 0) bif++;
			if (a == 0 && b == 1 && c == 0 && d == 0 && e == 1 && f == 1 && g == 1 && h == 0 && i == 0) bif++;
			if (a == 0 && b == 0 && c == 1 && d == 1 && e == 1 && f == 0 && g == 0 && h == 0 && i == 1) bif++;
			if (a == 1 && b == 0 && c == 0 && d == 0 && e == 1 && f == 1 && g == 0 && h == 1 && i == 0) bif++;
			if (a == 0 && b == 1 && c == 0 && d == 0 && e == 1 && f == 0 && g == 1 && h == 0 && i == 1) bif++;
			if (a == 0 && b == 0 && c == 1 && d == 1 && e == 1 && f == 0 && g == 0 && h == 1 && i == 0) bif++;

			if (a == 0 && b == 1 && c == 0 && d == 1 && e == 1 && f == 0 && g == 0 && h == 0 && i == 1) bif++;
			if (a == 0 && b == 1 && c == 0 && d == 0 && e == 1 && f == 1 && g == 1 && h == 0 && i == 0) bif++;
			if (a == 0 && b == 0 && c == 1 && d == 1 && e == 1 && f == 0 && g == 0 && h == 1 && i == 0) bif++;
			if (a == 1 && b == 0 && c == 0 && d == 0 && e == 1 && f == 1 && g == 0 && h == 1 && i == 0) bif++;

			float dx = vec[ind].first;
			float dy = vec[ind].second;
			/*    해당 Block의 방향대로 theta를 설정해줌*/
			float s1 = dy / (dx + FLT_EPSILON);
			unsigned char num;
			if (s1 == 0)num = 0;
			else if (s1 > 1)num = '2';
			else if (s1 > 0)num = '1';
			else if (s1 < -1)num = '2';
			else if (s1 < 0)num = '1';

			if ((ending))
			{
				Point pt = Point(y, x);
				circle(color, pt, 4, CV_RGB(0, 255, 0), 1, 8, 0);
				circle(imgt, pt, 4, CV_RGB(255, 255, 255), 1, 8, 0);
				T[SP] = 1;
				X[SP] = y;
				Y[SP] = x;
				O[SP] = num;
				SP++;
				ending = 0;

				for (int i = -2; i < 3; i++) {
					if ((x + i) < 0)continue;

					if (y >= 0)
						visit[y][x + i] = 1;
					if (y + 1 >= 0)
						visit[y + 1][x + i] = 1;
					if (y + 2 >= 0)
						visit[y + 2][x + i] = 1;
					if (y - 2 >= 0)
						visit[y - 2][x + i] = 1;
					if (y - 1 >= 0)
						visit[y - 1][x + i] = 1;


				}
				x = x + 2;
				endingcheck++;
			}

			if ((bif))
			{
				Point pt1 = Point(y - 2, x - 2);
				Point pt2 = Point(y + 2, x + 2);
				rectangle(color, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
				rectangle(imgt, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
				T[SP] = 3;
				X[SP] = y;
				Y[SP] = x;
				O[SP] = num;
				SP++;
				bif = 0;
				x = x + 2;
				bifcheck++;
			}

			a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0;

		}
	}
}
pair<Mat, vector<pair<float, float>>> find_core_delta(Mat src, int size, int SP, int X[], int Y[], unsigned char O[], unsigned char T[], int& core_check, int& delta_check) {

	Mat inputImage = src;

	inputImage.convertTo(inputImage, CV_32F, 1.0 / 255, 0);

	medianBlur(inputImage, inputImage, 3);

	int blockSize = size;
	Mat coredeltaPrint = inputImage.clone();
	cvtColor(coredeltaPrint, coredeltaPrint, COLOR_GRAY2BGR);

	Mat tmp(inputImage.size(), inputImage.type());
	Mat coherence(inputImage.size(), inputImage.type());
	Mat orientationMap = tmp.clone();

	Mat grad_x, grad_y;

	Sobel(inputImage, grad_x, inputImage.depth(), 1, 0, 3);
	Sobel(inputImage, grad_y, inputImage.depth(), 0, 1, 3);

	Mat Fx(inputImage.size(), inputImage.type()),
		Fy(inputImage.size(), inputImage.type()),
		Fx_gauss,
		Fy_gauss;
	Mat copy_input(inputImage.size(), inputImage.type());

	int width = inputImage.cols;
	int height = inputImage.rows;
	int blockH;
	int blockW;
	cout << "width" << width << endl;
	cout << "height" << height << endl;
	vector<pair<float, float>> vec;
	vector<int> cnt;
	for (int i = 0; i < height; i += blockSize) {
		for (int j = 0; j < width; j += blockSize) {
			float Gsx = 0.0;
			float Gsy = 0.0;
			float Gxx = 0.0;
			float Gyy = 0.0;

			//for check bounds of img
			blockH = ((height - i) < blockSize) ? (height - i) : blockSize;
			blockW = ((width - j) < blockSize) ? (width - j) : blockSize;

			//average at block WхW
			for (int u = i; u < i + blockH; u++) {
				for (int v = j; v < j + blockW; v++) {
					Gsx += (grad_x.at<float>(u, v) * grad_x.at<float>(u, v)) - (grad_y.at<float>(u, v) * grad_y.at<float>(u, v));
					Gsy += 2 * grad_x.at<float>(u, v) * grad_y.at<float>(u, v);
					Gxx += grad_x.at<float>(u, v) * grad_x.at<float>(u, v);
					Gyy += grad_y.at<float>(u, v) * grad_y.at<float>(u, v);
				}
			}

			float coh = sqrt(pow(Gsx, 2) + pow(Gsy, 2)) / (Gxx + Gyy);
			//copy_input
			float fi = 0.5f * fastAtan2(Gsy, Gsx) * CV_PI / 180.0f;

			Fx.at<float>(i, j) = cos(2 * fi);
			Fy.at<float>(i, j) = sin(2 * fi);

			//fill blocks
			for (int u = i; u < i + blockH; u++) {
				for (int v = j; v < j + blockW; v++) {
					orientationMap.at<float>(u, v) = fi;
					Fx.at<float>(u, v) = Fx.at<float>(i, j);
					Fy.at<float>(u, v) = Fy.at<float>(i, j);
					coherence.at<float>(u, v) = (coh < 0.85f) ? 1.0f : 0.0f;
				}
			}
		}
	}

	GaussianBlur(Fx, Fx_gauss, Size(5, 5), 1, 1);
	GaussianBlur(Fy, Fy_gauss, Size(5, 5), 1, 1);


	vector<vector<float>> vec_gradgrad(height, vector<float>(width, 0.0f));

	for (int m = 0; m < height; m++) {

		for (int n = 0; n < width; n++) {

			copy_input.at<float>(m, n) = 0.5f * fastAtan2(Fy_gauss.at<float>(m, n), Fx_gauss.at<float>(m, n)) * CV_PI / 180.0f;
			if ((m % blockSize) == 0 && (n % blockSize) == 0) {
				int x = n;
				int y = m;
				int ln = sqrt(2 * pow(blockSize, 2)) / 2;
				float dx = ln * cos(copy_input.at<float>(m, n) - CV_PI / 2.0f);
				float dy = ln * sin(copy_input.at<float>(m, n) - CV_PI / 2.0f);
				vec.push_back({ dx,dy });

				float grad = dy / (dx + FLT_EPSILON);

				float gradgrad = grad;

			
				if (2.0f <= gradgrad)
					gradgrad = FLT_MAX;
				else if (0.5f <= gradgrad && gradgrad < 2.0f)
					gradgrad = 1.0f;
				else if (-0.5f <= gradgrad && gradgrad < 0.5f)
					gradgrad = 0.0f;
				else if (-2.0f <= gradgrad && gradgrad < -0.5f)
					gradgrad = -1.0f;
				else if (gradgrad < -2.0f)
					gradgrad = FLT_MAX;

				vec_gradgrad[m][n] = gradgrad;
			}
		}
	}
	priority_queue<pair<int, pair<int, int>>> pq_core;
	priority_queue<pair<int, pair<int, int>>> pq_delta;

	for (int m = blockSize; m < height; m = m + blockSize) {
		for (int n = blockSize; n < width; n = n + blockSize) {

			if (SP == 50)break;
			if (vec_gradgrad[m][n - blockSize] == -1.0f && vec_gradgrad[m][n] == 0.0f && vec_gradgrad[m][n + blockSize] == 1.0f) {
				int up_up = 0;
				int up_side = 0;
				int down_up = 0;
				int down_side = 0;


				if (vec_gradgrad[m - blockSize][n - blockSize] == FLT_MAX)
					up_up += 2;
				else if (vec_gradgrad[m - blockSize][n - blockSize] == 1.0f || vec_gradgrad[m - blockSize][n - blockSize] == -1.0f) {
					up_up++;
					up_side++;
				}
				else if (vec_gradgrad[m - blockSize][n - blockSize] == 0.0f)
					up_side += 2;


				if (vec_gradgrad[m - blockSize][n] == FLT_MAX)
					up_up += 2;
				else if (vec_gradgrad[m - blockSize][n] == 1.0f || vec_gradgrad[m - blockSize][n] == -1.0f) {
					up_up++;
					up_side++;
				}
				else if (vec_gradgrad[m - blockSize][n] == 0.0f)
					up_side += 2;


				if (vec_gradgrad[m - blockSize][n + blockSize] == FLT_MAX)
					up_up += 2;
				else if (vec_gradgrad[m - blockSize][n + blockSize] == 1.0f || vec_gradgrad[m - blockSize][n + blockSize] == -1.0f) {
					up_up++;
					up_side++;
				}
				else if (vec_gradgrad[m - blockSize][n + blockSize] == 0.0f)
					up_side += 2;



				if (vec_gradgrad[m + blockSize][n - blockSize] == FLT_MAX)
					down_up += 2;
				else if (vec_gradgrad[m + blockSize][n - blockSize] == 1.0f || vec_gradgrad[m + blockSize][n - blockSize] == -1.0f) {
					down_up++;
					down_side++;
				}
				else if (vec_gradgrad[m + blockSize][n - blockSize] == 0.0f)
					down_side += 2;

				if (vec_gradgrad[m + blockSize][n] == FLT_MAX)
					down_up += 2;
				else if (vec_gradgrad[m + blockSize][n] == 1.0f || vec_gradgrad[m + blockSize][n] == -1.0f) {
					down_up++;
					down_side++;
				}
				else if (vec_gradgrad[m + blockSize][n] == 0.0f)
					down_side += 2;


				if (vec_gradgrad[m + blockSize][n + blockSize] == FLT_MAX)
					down_up += 2;
				else if (vec_gradgrad[m + blockSize][n + blockSize] == 1.0f || vec_gradgrad[m + blockSize][n + blockSize] == -1.0f) {
					down_up++;
					down_side++;
				}
				else if (vec_gradgrad[m + blockSize][n + blockSize] == 0.0f)
					down_side += 2;

				
				int cnt_core = up_side + down_up;
				int cnt_delta = up_up + down_side;

				// 차이가 2 넘는 경우만 push, 코어가 더 많으면 코어, 델타가 더 많으면 델타
				if (abs(cnt_delta - cnt_core) > 2) {
					if (cnt_delta <= cnt_core) {
						pq_core.push({ cnt_core - cnt_delta,{ m, n } });
						T[SP] = 10;
					}
					else {
						pq_delta.push({ cnt_delta - cnt_core,{ m, n } });
						T[SP] = 11;
					}
					X[SP] = m;
					Y[SP] = n;
					O[SP] = 0;
					SP++;
				}
			}


		}
	}



	if (!pq_core.empty()) {
		circle(coredeltaPrint, Point(pq_core.top().second.second + blockSize / 2, pq_core.top().second.first + blockSize / 2), 5, Scalar(0, 0, 255), 1, 16);
		core_check++;
	}

	if (!pq_delta.empty()) {
		int mmm = pq_delta.top().second.first;
		int nnn = pq_delta.top().second.second;

		line(coredeltaPrint, Point(nnn + blockSize / 2, mmm), Point(nnn + blockSize / 2, mmm + blockSize), Scalar(0, 0, 255), 1, LINE_AA, 0);
		line(coredeltaPrint, Point(nnn, mmm + blockSize / 2), Point(nnn + blockSize, mmm + blockSize / 2), Scalar(0, 0, 255), 1, LINE_AA, 0);
		delta_check++;
		pq_delta.pop();
	}
	if (!pq_delta.empty()) {
		int mmm = pq_delta.top().second.first;
		int nnn = pq_delta.top().second.second;
		line(coredeltaPrint, Point(nnn + blockSize / 2, mmm), Point(nnn + blockSize / 2, mmm + blockSize), Scalar(0, 0, 255), 1, LINE_AA, 0);
		line(coredeltaPrint, Point(nnn, mmm + blockSize / 2), Point(nnn + blockSize, mmm + blockSize / 2), Scalar(0, 0, 255), 1, LINE_AA, 0);

	}


	normalize(orientationMap, orientationMap, 0, 1, NORM_MINMAX);

	orientationMap = copy_input.clone();

	normalize(copy_input, copy_input, 0, 1, NORM_MINMAX);

	pyrUp(copy_input, copy_input);

	pair<Mat, vector<pair<float, float>>> returning;

	returning = { coredeltaPrint, vec };
	return returning;


}
void get_rid_of(Mat& imgt)
{

	for (int y = 2; y < (imgt.rows) - 2; y++)
	{
		for (int x = 2; x < (imgt.cols) - 2; x++)
		{
			int pix1 = imgt.at<uchar>(y - 2, x - 2);
			int pix2 = imgt.at<uchar>(y - 2, x - 1);
			int pix3 = imgt.at<uchar>(y - 2, x);
			int pix4 = imgt.at<uchar>(y - 2, x + 1);
			int pix5 = imgt.at<uchar>(y - 2, x + 2);

			int pix6 = imgt.at<uchar>(y - 1, x - 2);
			int pix7 = imgt.at<uchar>(y - 1, x - 1);
			int pix8 = imgt.at<uchar>(y - 1, x);
			int pix9 = imgt.at<uchar>(y - 1, x + 1);
			int pix10 = imgt.at<uchar>(y - 1, x + 2);

			int pix11 = imgt.at<uchar>(y, x - 2);
			int pix12 = imgt.at<uchar>(y, x - 1);
			int pix13 = imgt.at<uchar>(y, x);
			int pix14 = imgt.at<uchar>(y, x + 1);
			int pix15 = imgt.at<uchar>(y, x + 2);

			int pix16 = imgt.at<uchar>(y + 1, x - 2);
			int pix17 = imgt.at<uchar>(y + 1, x - 1);
			int pix18 = imgt.at<uchar>(y + 1, x);
			int pix19 = imgt.at<uchar>(y + 1, x + 1);
			int pix20 = imgt.at<uchar>(y + 1, x + 2);

			int pix21 = imgt.at<uchar>(y + 2, x - 2);
			int pix22 = imgt.at<uchar>(y + 2, x - 1);
			int pix23 = imgt.at<uchar>(y + 2, x);
			int pix24 = imgt.at<uchar>(y + 2, x + 1);
			int pix25 = imgt.at<uchar>(y + 2, x + 2);


			if (pix1 == 0 && pix2 == 0 && pix3 == 0 && pix4 == 0 && pix5 == 0 && pix6 == 0 && pix10 == 0 && pix11 == 0 && pix15 == 0 && pix16 == 0 && pix20 == 0 && pix21 == 0 && pix22 == 0 && pix23 == 0 && pix24 == 0 && pix25 == 0 && (pix7 > 0 || pix8 > 0 || pix9 > 0 || pix12 > 0 || pix13 > 0 || pix14 > 0 || pix17 > 0 || pix18 > 0 || pix19 > 0)) {
				imgt.at<uchar>(y - 1, x - 1) = 0;
				imgt.at<uchar>(y - 1, x) = 0;
				imgt.at<uchar>(y - 1, x + 1) = 0;

				imgt.at<uchar>(y, x - 1) = 0;
				imgt.at<uchar>(y, x) = 0;
				imgt.at<uchar>(y, x + 1) = 0;

				imgt.at<uchar>(y + 1, x - 1) = 0;
				imgt.at<uchar>(y + 1, x) = 0;
				imgt.at<uchar>(y + 1, x + 1) = 0;

			}
		}
	}
}
int main() {
	int block_size = 7;
	int W = 152;                     
	int H = 200;                     
	int M = 0;                        
	int SP = 0;                       
	int X[100];                  
	int Y[100];                   
	unsigned char O[100];          
	unsigned char T[100];                    
	ofstream output("[40_Im]_2019_5_1_R_T#1.bin", ios::out | ios::binary);
	for (int i = 0; i < 50; i++) {
		X[i] = 0; Y[i] = 0; O[i] = 0; T[i] = 0;
	}
	Mat src = imread("C:\\Users\\JungYeonHee\\source\\repos\\DIP_3\\DIP_3\\imput\\[40_Im]_2019_5_1_R_T#1.bmp");
	cvtColor(src, src, COLOR_RGB2GRAY);
	Mat show;
	equalizeHist(src, show);
	Mat dil;
	Mat ero;

	dilate(show, dil, Mat(), Point(-1, -1));
	erode(show, ero, Mat(), Point(-1, -1));
	Mat open, res;
	open = dil - ero;//검은색 흰색이 반전되어 있어서 dil-ero를 해야 opening 효과 냄
	erode(open, ero, Mat(), Point(-1, -1));//opening한걸 침식
	dilate(open, dil, Mat(), Point(-1, -1));//opening한걸 팽창

	res = dil - ero;

	int A, B, C, D, E;
	int AA[50], BB[50];
	unsigned char OOO[50], TTT[50];
	int core_check = 0;
	int delta_check = 0;

	pair<Mat, vector<pair<float, float>>> returned = orientation(show, block_size, &A, AA, BB, OOO, TTT, core_check, delta_check);
	Mat orientationmap = returned.first;
	vector<pair<float, float>> vec = returned.second;

	Mat segmented;
	Mat segmented2 = segmentation(src, segmented);
	Mat plus;
	threshold(segmented, plus, 127, 255, THRESH_BINARY_INV);
	Mat gabored = gabor(res, vec, block_size) - plus;//태두리 노이즈 제거
	pyrUp(src, src);
	pyrUp(show, show);
	pyrUp(segmented2, segmented2);
	pyrUp(orientationmap, orientationmap);
	pair<Mat, vector<pair<float, float>>> returned2 = find_core_delta(gabored, block_size, SP, X, Y, O, T, core_check, delta_check);
	Mat find_single = returned2.first;
	M = SP;
	pyrUp(gabored, gabored);

	Mat gabored_end;

	threshold(gabored, gabored_end, 1, 255, THRESH_BINARY_INV);
	//// thinning
	Mat imgt = thinning(gabored);
	threshold(gabored_end, gabored_end, 1, 255, THRESH_BINARY_INV);
	get_rid_of(imgt);
	threshold(imgt, imgt, 1, 255, THRESH_BINARY_INV);
	imgt.convertTo(imgt, CV_8U);
	imshow("thinned", imgt);

	threshold(imgt, imgt, 1, 255, THRESH_BINARY_INV);

	pyrUp(find_single, find_single);
	Mat color = find_single;
	int bifcheck = 0;
	int endingcheck = 0;
	find_other_maniture(color, imgt, endingcheck, bifcheck, SP, X, Y, O, T, vec);
	imshow("result", color);
	cout << "ending : " << endingcheck << endl;
	cout << "bif : " << bifcheck << endl;
	cout << "core :" << core_check << endl;
	cout << "delta :" << delta_check << endl;

	SP = delta_check + core_check;
	M = endingcheck + bifcheck + SP;
	cout << SP << " " << M;
	output.write((char*)&W, sizeof(int));
	output.write((char*)&H, sizeof(int));
	output.write((char*)&M, sizeof(int));
	output.write((char*)&SP, sizeof(int));
	for (int i = 0; i < 50; i++) {
		output.write((char*)&X[i], sizeof(int));
		output.write((char*)&Y[i], sizeof(int));
		output.write((char*)&O[i], sizeof(char));
		output.write((char*)&T[i], sizeof(char));
	}

	cv::waitKey(0);
}
