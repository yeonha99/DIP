#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;
void get_persentage(Mat const& src) {//��ġ�ϴ� �ȼ��� ��ü�ȼ��� ���
	long long equal=0;

	long long all = src.rows * src.cols;//��ü�ȼ��� ���
	for (int i = 0; i < src.rows; i++) {//��ġ�ϴ� �ȼ��� ���
		for (int j = 0; j < src.cols; j++) {
			if (src.ptr<uchar>(i)[j] != 0) {//0�� �ƴ϶�� �� ��ġ �� ���
				equal++;//��ġ�� ����
			}

		}
	}
	cout << "��ġ�ȼ��� : " << equal << endl;
	cout << "��ü�ȼ��� : " <<all << endl;
}
int main()
{
	Mat frame1, prvs;
	frame1= imread("C:/Users/JungYeonHee/source/repos/DIP_5/DIP_5/optical flow_test/Backyard/a.PNG");//�̹��� 1 �ε�
	medianBlur(frame1, frame1,3);//������ ����
	cvtColor(frame1, prvs, COLOR_BGR2GRAY);//graylevel�� ��ȯ
	equalizeHist(prvs, prvs);//H.E�� ���Ͽ� ��ó��
		Mat frame2, next;
		Mat result;
		frame2=imread("C:/Users/JungYeonHee/source/repos/DIP_5/DIP_5/optical flow_test/Backyard/b.PNG");//�̹��� 2 �ε� 
		medianBlur(frame2, frame2, 3);//������ ����
		cvtColor(frame2, next, COLOR_BGR2GRAY);//graylevel�� ��ȯ
		equalizeHist(next, next);//H.E�� ���Ͽ� ��ó��
			Mat flow(prvs.size(), CV_32FC2);
			calcOpticalFlowFarneback(prvs, next, flow, 0.1, 3, 20, 3, 7, 1.5, 0);
			Mat flow_parts[2];
			split(flow, flow_parts);
			Mat magnitude, angle, magn_norm;
			cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
			normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);//graylevel�� ����� ���ϱ� ������ magnitude�� ����ϸ� �ȴ�.
			Mat ok;
			Mat compar;
			magn_norm.convertTo(magn_norm, CV_8U, 255.0);//������ graylevel�� ��ȯ
			imshow("magn_norm", magn_norm);
			ok = imread("C:/Users/JungYeonHee/source/repos/DIP_5/DIP_5/optical flow_test/Backyard/flowGT.PNG");
			cvtColor(ok, ok, COLOR_BGR2GRAY);//compare�Լ��� ����ϱ� ���ؼ� graylevel�� ��ȯ
			compare(ok, magn_norm, compar, CMP_EQ);//cmp_eq�� ���ؼ� ��ġ�� ��� 255�� ��ġ���� ���� ��� 0���� �Ҵ�
			imshow("compar", compar);
			get_persentage(compar);//compare ��� �̹������� ��ġ �ȼ� �� ���
	
			waitKey(0);

}