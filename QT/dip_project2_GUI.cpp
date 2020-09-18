#include "dip_project2_GUI.h"
#include <Windows.h>
#include <fileapi.h>
#include <iostream>
#include <string>
#include <string.h>
#include <vector>
#include <QPushButton>
#include <qmessagebox.h>
#include <qfiledialog.h>
#include <qlabel.h>
#include <qpixmap.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

QFileInfoList fileinfolist;
QFileInfoList fileinfolist2;
QString save_lot;
QString result_lot;
int ind = 0;//불러온 폴더에 접근하기 위한 index
int ind2 = 0;//결과 사진을 저장한 폴더에 접근하기 위한 index
double ticks = getTickFrequency();
int64 t0;

vector<double> time_save;//감지시간을 저장하기 위한 vector


dip_project2_GUI::dip_project2_GUI(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
}

void dip_project2_GUI::openButton() {//가져올 폴더를 여는 함수
	QFileDialog dialog(this);
	QDir dir;
	dialog.setFileMode(QFileDialog::Directory);
	save_lot = QFileDialog::getExistingDirectory();
	ui.labelPath->setText(save_lot);
	dir.setPath(save_lot);
	QStringList filters;
	filters << "*.png" << "*.jpg" << "*.bmp";
	fileinfolist.clear();
	fileinfolist = dir.entryInfoList(filters, QDir::Files | QDir::NoDotAndDotDot);
	ui.label_fileSize->setText(QString::number(fileinfolist.length()));
	ui.list->clear();
	for (auto file : fileinfolist)//비어있을 때 까지 list에 파일의 이름을 넣어준다.
		ui.list->addItem(file.fileName());
	displayImage();
}

void dip_project2_GUI::displayImage() {//선택한 이미지를 보여주는 함수
	QString img_path = save_lot + "/" + fileinfolist[ind].fileName();//경로를 위한 string 변수
	QImage img(img_path);
	QPixmap buf = QPixmap::fromImage(img.scaled(ui.labelImage->width(), ui.labelImage->height()));
	ui.labelImage->setPixmap(buf);
	ui.labelImage->resize(buf.width(), buf.height());
}

void dip_project2_GUI::displayImage2() {//결과 이미지를 보여주는 함수
	QString img_path = result_lot + "/" + fileinfolist2[ind2].fileName();
	QImage img(img_path);
	QPixmap buf = QPixmap::fromImage(img.scaled(ui.label->width(), ui.label->height()));
	ui.label->setPixmap(buf);
	ui.label->resize(buf.width(), buf.height());
	ui.time->setText(QString::number(time_save[ind2]));
}
void dip_project2_GUI::selectedImage() {//해당 이미지를 선택시 index 갱신
	ind = ui.list->currentRow();
	displayImage();
}
void dip_project2_GUI::selectedImage2() {//해당 이미지를 선택시 index 갱신
	ind2 = ui.resultlist->currentRow();
	displayImage2();
}
void dip_project2_GUI::processButton() {//실행을 눌렀을 경우
	bool check_face = ui.checkBox->isChecked();//어떤 checkbox를 선택했는지에 따라서 행동하게 변수로 저장

	bool check_car = ui.checkBox_2->isChecked();

	QDir source, resultD;
	source.setPath(save_lot);
	resultD = source;
	resultD.cdUp();
	string img_dir = save_lot.toStdString() + '/';
	string result_dir = resultD.absolutePath().toStdString() + "/result";
	wstring temp = wstring(result_dir.begin(), result_dir.end());
	CreateDirectory(temp.c_str(), NULL);
	result_lot = result_lot.fromStdString(result_dir);
	resultD.setPath(result_lot);
	ui.label_3->setText(result_lot);//저장 경로 저장

	result_dir += '/';
	int count = 0;
	for (auto file : fileinfolist) {
		count++;

		string img_path = img_dir + file.fileName().toStdString();
		Mat img = imread(img_path);

		vector<Rect> faces;
		vector<Rect> cars;

		if (check_face) {//face 박스에 선택시 작동
			CascadeClassifier face_cascade;
			face_cascade.load("D:\\DIP\\face\\90_20 change data\\cascade.xml");
			t0 = getTickCount();
			face_cascade.detectMultiScale(img, faces, 1.3, 3, 0 | CASCADE_SCALE_IMAGE);
			t0 = getTickCount() - t0;

			for (int i = 0; i < faces.size(); i++)//픽셀에 접근해서 판별
				rectangle(img, faces[i], Scalar(255, 0, 0), 2, 1);
		}

		if (check_car) {//car 박스에 선택시 작동
			CascadeClassifier car_cascade;
			car_cascade.load("D:\\DIP\\car\\car_8015.xml");
			t0 = getTickCount();
			cv::resize(img, img, Size(320, 250));
			car_cascade.detectMultiScale(img, cars, 1.1, 5, 0 | CASCADE_SCALE_IMAGE);
			t0 = getTickCount() - t0;

			for (int i = 0; i < cars.size(); i++)//픽셀에 접근해서 판별
				rectangle(img, cars[i], Scalar(0, 0, 255), 2, 1);
		}

		double time = ((double)t0 * 1000) / ticks; // ms단위로 측정
		time_save.push_back(time);//이미지마다 걸린 시간을 저장하기 위해서 vector에 저장한다.

		imwrite(result_dir + to_string(time) + ".jpg", img);//결과 이미지 저장
	}
	QMessageBox msg;
	msg.setText("end!");//test 종료 알림
	msg.exec();

	QStringList filters2;
	filters2 << "*.png" << "*.jpg" << "*.bmp";
	fileinfolist2.clear();
	fileinfolist2 = resultD.entryInfoList(filters2, QDir::Files | QDir::NoDotAndDotDot);
	ui.resultlist->clear();
	for (auto file : fileinfolist2)//결과 리스트에 사진 이미지 push
		ui.resultlist->addItem(file.fileName());

	displayImage2();//결과 이미지를 보여주기 위해 함수 리콜
}