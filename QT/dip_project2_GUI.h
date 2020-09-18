#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_dip_project2_GUI.h"
#include <Windows.h>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

class dip_project2_GUI : public QMainWindow
{
	Q_OBJECT

public:
	dip_project2_GUI(QWidget* parent = Q_NULLPTR);


private:
	Ui::dip_project2_GUIClass ui;
private slots:
	void openButton();
	void displayImage();
	void selectedImage();
	void displayImage2();
	void selectedImage2();
	void processButton();
};