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
int ind = 0;//�ҷ��� ������ �����ϱ� ���� index
int ind2 = 0;//��� ������ ������ ������ �����ϱ� ���� index
double ticks = getTickFrequency();
int64 t0;

vector<double> time_save;//�����ð��� �����ϱ� ���� vector


dip_project2_GUI::dip_project2_GUI(QWidget* parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
}

void dip_project2_GUI::openButton() {//������ ������ ���� �Լ�
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
	for (auto file : fileinfolist)//������� �� ���� list�� ������ �̸��� �־��ش�.
		ui.list->addItem(file.fileName());
	displayImage();
}

void dip_project2_GUI::displayImage() {//������ �̹����� �����ִ� �Լ�
	QString img_path = save_lot + "/" + fileinfolist[ind].fileName();//��θ� ���� string ����
	QImage img(img_path);
	QPixmap buf = QPixmap::fromImage(img.scaled(ui.labelImage->width(), ui.labelImage->height()));
	ui.labelImage->setPixmap(buf);
	ui.labelImage->resize(buf.width(), buf.height());
}

void dip_project2_GUI::displayImage2() {//��� �̹����� �����ִ� �Լ�
	QString img_path = result_lot + "/" + fileinfolist2[ind2].fileName();
	QImage img(img_path);
	QPixmap buf = QPixmap::fromImage(img.scaled(ui.label->width(), ui.label->height()));
	ui.label->setPixmap(buf);
	ui.label->resize(buf.width(), buf.height());
	ui.time->setText(QString::number(time_save[ind2]));
}
void dip_project2_GUI::selectedImage() {//�ش� �̹����� ���ý� index ����
	ind = ui.list->currentRow();
	displayImage();
}
void dip_project2_GUI::selectedImage2() {//�ش� �̹����� ���ý� index ����
	ind2 = ui.resultlist->currentRow();
	displayImage2();
}
void dip_project2_GUI::processButton() {//������ ������ ���
	bool check_face = ui.checkBox->isChecked();//� checkbox�� �����ߴ����� ���� �ൿ�ϰ� ������ ����

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
	ui.label_3->setText(result_lot);//���� ��� ����

	result_dir += '/';
	int count = 0;
	for (auto file : fileinfolist) {
		count++;

		string img_path = img_dir + file.fileName().toStdString();
		Mat img = imread(img_path);

		vector<Rect> faces;
		vector<Rect> cars;

		if (check_face) {//face �ڽ��� ���ý� �۵�
			CascadeClassifier face_cascade;
			face_cascade.load("D:\\DIP\\face\\90_20 change data\\cascade.xml");
			t0 = getTickCount();
			face_cascade.detectMultiScale(img, faces, 1.3, 3, 0 | CASCADE_SCALE_IMAGE);
			t0 = getTickCount() - t0;

			for (int i = 0; i < faces.size(); i++)//�ȼ��� �����ؼ� �Ǻ�
				rectangle(img, faces[i], Scalar(255, 0, 0), 2, 1);
		}

		if (check_car) {//car �ڽ��� ���ý� �۵�
			CascadeClassifier car_cascade;
			car_cascade.load("D:\\DIP\\car\\car_8015.xml");
			t0 = getTickCount();
			cv::resize(img, img, Size(320, 250));
			car_cascade.detectMultiScale(img, cars, 1.1, 5, 0 | CASCADE_SCALE_IMAGE);
			t0 = getTickCount() - t0;

			for (int i = 0; i < cars.size(); i++)//�ȼ��� �����ؼ� �Ǻ�
				rectangle(img, cars[i], Scalar(0, 0, 255), 2, 1);
		}

		double time = ((double)t0 * 1000) / ticks; // ms������ ����
		time_save.push_back(time);//�̹������� �ɸ� �ð��� �����ϱ� ���ؼ� vector�� �����Ѵ�.

		imwrite(result_dir + to_string(time) + ".jpg", img);//��� �̹��� ����
	}
	QMessageBox msg;
	msg.setText("end!");//test ���� �˸�
	msg.exec();

	QStringList filters2;
	filters2 << "*.png" << "*.jpg" << "*.bmp";
	fileinfolist2.clear();
	fileinfolist2 = resultD.entryInfoList(filters2, QDir::Files | QDir::NoDotAndDotDot);
	ui.resultlist->clear();
	for (auto file : fileinfolist2)//��� ����Ʈ�� ���� �̹��� push
		ui.resultlist->addItem(file.fileName());

	displayImage2();//��� �̹����� �����ֱ� ���� �Լ� ����
}