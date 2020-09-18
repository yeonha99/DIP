#include "dip_project2_GUI.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    dip_project2_GUI w;
   
    w.show();
    return a.exec();
}
