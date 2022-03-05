#include <iostream>
#include <string>
#include <sstream>
#include <cmath>

// OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"

cv::Mat light_pat;
cv::Ptr<cv::ml::SVM> svm;

const char* keys =
{
	"{help h usage ? | | print this message}"
	"{@image || Image to classify}"
};


static cv::Scalar randomColor(cv::RNG& rng)
{
	int icolor = (unsigned)rng;
	return cv::Scalar(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
}

cv::Mat removeLight(cv::Mat img, cv::Mat pattern)
{
	cv::Mat aux;

	// Require change our image to 32 float for division
	cv::Mat img32, pattern32;
	img.convertTo(img32, CV_32F);
	pattern.convertTo(pattern32, CV_32F);
	// Divide the imabe by the pattern
	aux = 1 - img32 / pattern32;
	// Scale it to convert o 8bit format
	aux = aux * 255;
	// Convert 8 bits format
	aux.convertTo(aux, CV_8U);

	//equalizeHist( aux, aux );
	return aux;
}





cv::Mat preprocessImg(cv::Mat inp)
{
    cv::Mat res;
    cv::Mat img_denois, img_box_smooth;
    cv::medianBlur(inp,img_denois, 3);
    
    cv::Mat img_no_ligh;
    img_denois.copyTo(img_no_ligh);
    cv::blur(img_denois,light_pat,cv::Size(img_denois.cols/3,img_denois.rows/3));
    img_no_ligh = removeLight(img_denois,light_pat);

    


    cv::threshold(img_no_ligh,res, 30, 255, cv::THRESH_BINARY);
    
    return res;
}

std::vector< std::vector<float> >  ExtractFeatures(cv::Mat img,
                                                    std::vector<int> * left = NULL,
                                                    std::vector<int> * top = NULL)
{
    std::vector< std::vector<float> > out;
    std::vector< std::vector<cv::Point> > countours;
    std::vector<cv::Vec4i>  hierarchy;

    cv::Mat inp = img.clone();

    cv::findContours(inp, countours,hierarchy,cv::RETR_CCOMP,cv::CHAIN_APPROX_SIMPLE);
    if (countours.size() == 0)
    {
        return out;
    }

    cv::RNG rng(0);
    for (int i = 0; i <countours.size(); i++)
    {
        cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
        cv::drawContours(mask,countours,i,cv::Scalar(1),cv::FILLED,cv::LINE_8,hierarchy,1);
        cv::Scalar area_s = sum(mask);
        float area = area_s[0];

        if (area > 500)
        {
            cv::RotatedRect r = cv::minAreaRect(countours[i]);
            float width = r.size.width;
            float height = r.size.height;
            float ar = (width<height) ? height/width : width/height;

            std::vector<float> row;
            row.push_back(area);
            row.push_back(ar);
            out.push_back(row);
            if(left != NULL)
            {
                left->push_back((int) r.center.x);
            }
            if (top != NULL)
            {
                top->push_back((int) r.center.y);
            }
            
            cv::imshow("MASK",mask*255);
            cv::waitKey(1);
        }
    }
    return out;

}

bool readFolderAndExtractFeatures(std::string folder,int label, int nb_for_test, 
    std::vector<float> &data_train, std::vector<int> &lables_train,
    std::vector<float> &data_test,std::vector<float> &labels_test)
{
    cv::VideoCapture imgs;
    if (!imgs.open(folder))
    {
        std::cout<<"Can not Open the Folder images"<<std::endl;
        return false;

    }
    cv::Mat frame;
    int img_index = 0;
    while(imgs.read(frame))
    {
        cv::cvtColor(frame,frame,cv::COLOR_BGR2GRAY);
        frame = preprocessImg(frame);
        std::vector<std::vector<float>> features = ExtractFeatures(frame);
        for(int i =0; i <features.size();i++)
        {
            if (img_index>=nb_for_test)
            {
                data_train.push_back(features[i][0]);
                data_train.push_back(features[i][1]);
                lables_train.push_back(label);
            }
            else 
            {
                data_test.push_back(features[i][0]);
                data_test.push_back(features[i][1]);
                labels_test.push_back((float)label);
                
            }
        }
        img_index++;
    }
    return true;
}



int main (int argc, const char ** argv)
{
    std::vector<float> data_train;
    std::vector<int> lables_train;
    std::vector<float> data_test;
    std::vector<float> labels_test;

    int nb_for_test = 20;

 

    readFolderAndExtractFeatures("../../data/nut/tuerca_%04d.pgm", 0, nb_for_test, data_train, lables_train, data_test, labels_test);
	// Get and process the ring images
	readFolderAndExtractFeatures("../../data/ring/arandela_%04d.pgm", 1, nb_for_test, data_train, lables_train, data_test, labels_test);
	// get and process the screw images
	readFolderAndExtractFeatures("../../data/screw/tornillo_%04d.pgm", 2, nb_for_test, data_train, lables_train, data_test, labels_test);

    std::cout << "Num of train samples: " << data_train.size() << std::endl;

	std::cout << "Num of test samples: " << labels_test.size() << std::endl;

    cv::Mat mat_data_train(data_train.size()/2,2,CV_32FC1,&data_train[0]);
    cv::Mat mat_labels_train(lables_train.size(),1,CV_32SC1,&lables_train[0]);

    cv::Mat mat_data_test(data_test.size()/2,2,CV_32FC1,&data_test[0]);
    cv::Mat mat_labels_test(labels_test.size(),1,CV_32FC1,&labels_test[0]);

    
    /*
    Ptr<TrainData> create(InputArray samples, int layout, InputArray responses,
                                 InputArray varIdx=noArray(), InputArray sampleIdx=noArray(),
                                 InputArray sampleWeights=noArray(), InputArray varType=noArray());
                                 */
    // cv::Ptr<cv::ml::TrainData> train_data_ptr = cv::ml::TrainData::create(mat_data_train,cv::ml::ROW_SAMPLE,
    //                                                                 mat_labels_train);
    
    svm = cv::ml::SVM::create();
    std::cout<<"mat_data_train"<<mat_data_train.size() <<std::endl;
    std::cout<<"mat_labels_train"<<mat_labels_train.size() <<std::endl;
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    svm->setC(0.1);
    svm->setKernel(cv::ml::SVM::CHI2);
    // cv::ml::VariableTypes vt()
    // svm->train(train_data_pt)
    // svm->train(train_data_ptr,);

    svm->train(mat_data_train,cv::ml::ROW_SAMPLE,mat_labels_train);
    std::cout<<"here"<<std::endl;
    std::cout << "Evaluation" << std::endl;
	std::cout << "==========" << std::endl;

    cv::Mat test_perdict;

    svm->predict(mat_data_test,test_perdict);

    std::cout << "Prediction Done" << std::endl;

    cv::Mat error_mat = test_perdict!=mat_labels_test;
    float err = (100.0f*cv::countNonZero(error_mat) ) / labels_test.size();

    std::cout << "Error: " << err << "\%" << std::endl;







    return 0;
}







