#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;
void get_persentage(Mat const& src) {//일치하는 픽셀과 전체픽셀수 계산
	long long equal=0;

	long long all = src.rows * src.cols;//전체픽셀수 계산
	for (int i = 0; i < src.rows; i++) {//일치하는 픽셀수 계산
		for (int j = 0; j < src.cols; j++) {
			if (src.ptr<uchar>(i)[j] != 0) {//0이 아니라면 즉 일치 할 경우
				equal++;//일치수 증가
			}

		}
	}
	cout << "일치픽셀수 : " << equal << endl;
	cout << "전체픽셀수 : " <<all << endl;
}
int main()
{
	Mat frame1, prvs;
	frame1= imread("C:/Users/JungYeonHee/source/repos/DIP_5/DIP_5/optical flow_test/Backyard/a.PNG");//이미지 1 로드
	medianBlur(frame1, frame1,3);//노이즈 제거
	cvtColor(frame1, prvs, COLOR_BGR2GRAY);//graylevel로 변환
	equalizeHist(prvs, prvs);//H.E를 통하여 전처리
		Mat frame2, next;
		Mat result;
		frame2=imread("C:/Users/JungYeonHee/source/repos/DIP_5/DIP_5/optical flow_test/Backyard/b.PNG");//이미지 2 로드 
		medianBlur(frame2, frame2, 3);//노이즈 제거
		cvtColor(frame2, next, COLOR_BGR2GRAY);//graylevel로 변환
		equalizeHist(next, next);//H.E를 통하여 전처리
			Mat flow(prvs.size(), CV_32FC2);
			calcOpticalFlowFarneback(prvs, next, flow, 0.1, 3, 20, 3, 7, 1.5, 0);
			Mat flow_parts[2];
			split(flow, flow_parts);
			Mat magnitude, angle, magn_norm;
			cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
			normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);//graylevel의 결과를 원하기 때문에 magnitude를 사용하면 된다.
			Mat ok;
			Mat compar;
			magn_norm.convertTo(magn_norm, CV_8U, 255.0);//강도를 graylevel로 변환
			imshow("magn_norm", magn_norm);
			ok = imread("C:/Users/JungYeonHee/source/repos/DIP_5/DIP_5/optical flow_test/Backyard/flowGT.PNG");
			cvtColor(ok, ok, COLOR_BGR2GRAY);//compare함수를 사용하기 위해서 graylevel로 변환
			compare(ok, magn_norm, compar, CMP_EQ);//cmp_eq를 통해서 일치할 경우 255로 일치하지 않을 경우 0으로 할당
			imshow("compar", compar);
			get_persentage(compar);//compare 결과 이미지에서 일치 픽셀 수 계산
	
			waitKey(0);

}