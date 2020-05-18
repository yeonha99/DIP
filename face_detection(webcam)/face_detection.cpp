//#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/objdetect.hpp>
//#include <opencv2/imgproc.hpp>
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
////double ticks = cvGetTickFrequency()*10e6;
////int64 t0;
//
//int main(int argc, char** argv)
//{
//	// capture from web camera init
//// INPUT OF AN IMAGE
//	VideoCapture cap(0);
//	cap.open(0);
//
//	Mat img;
//
//	// Below mention YOUR cascade classifier instead of haarcascade_frontalface_alt2.xml file
//
//	CascadeClassifier face_cascade;
//	//face_cascade.load("Add the path from opencv sources/haarcascade_frontalface_alt2.xml"); // OpenCV GIVEN DEFAULT CLASSIFIER XML
//	face_cascade.load("cascade.xml"); // Trained own face detector
//
//
//	for (;;)
//	{
//
//		// Image from camera to Mat
//
//		cap >> img;
//
//		// obtain input image from source
//		cap.retrieve(img, CV_CAP_OPENNI_BGR_IMAGE);
//
//		// Just resize input image if you want
//		resize(img, img, Size(1000, 640));
//
//		// Container of faces
//		vector<Rect> faces;
//
//		//t0 = getTickCount();
//		// Detect faces
//		face_cascade.detectMultiScale(img, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(140, 140));
//		//t0 = getTickCount() - t0;
//		//printf("Processing Time : %.3f sec\n", (double)t0 / ticks);
//
//		// To draw rectangles around detected faces
//		for (unsigned i = 0; i < faces.size(); i++)
//			rectangle(img, faces[i], Scalar(255, 0, 0), 2, 1);
//
//
//		imshow("face", img);
//		int key2 = waitKey(20);
//
//	}
//	return 0;
//}





// EXTRA CODE

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

#define CAM_WIDTH 1280
#define CAM_HEIGHT 720

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name;
CascadeClassifier face_cascade;
String window_name = "Face detection";

/** @function main */
int main(int argc, const char** argv)
{
	face_cascade_name = "C:\\openCV4\\opencv-4.2.0\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading face cascade\n"); return -1; };

	VideoCapture cam(0);
	Mat frame;

	cam.set(CAP_PROP_FRAME_WIDTH, CAM_WIDTH);
	cam.set(CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);

	if (!cam.isOpened()) { printf("--(!)Error opening video cam\n"); return -1; }

	while (cam.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No camd frame -- Break!");
			break;
		}

		detectAndDisplay(frame);
		char c = (char)waitKey(10);
		if (c == 27) { break; }
	}
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2),
			0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
	}

	imshow(window_name, frame);
}
