#include "opencv2/highgui/highgui.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core_c.h>
#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <map>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

void drawLines(Mat& in, vector<Point> &points) {
    for (int i = 0; i < points.size(); i++)
        line(in, points[i], points[(i + 1) % points.size()], Scalar(200), 5);
}

void draw4Lines(Mat& img, Point2f points[]) {
    for (int i = 0; i < 4; i++)
        line(img, points[i], points[(i + 1) % 4], Scalar(100), 5);
}

void drawRotatedRect(Mat &img, RotatedRect &rect) {
    Point2f vertices[4];
    rect.points(vertices);

    draw4Lines(img, vertices);
}

bool invalidRect(Mat& img, RotatedRect& rect) {
    Point2f vertices[4];
    rect.points(vertices);

    for (int i = 0; i < 4; i++)
        if (vertices[i].x <= 0 || vertices[i].y <= 0 || vertices[i].x >= img.cols - 1 || vertices[i].y >= img.rows - 1)
            return true;

    return false;
}

// Предварительная проверка контура
bool sizePreCheck(const Size &imgSize, const RotatedRect& rect) {
    auto size = rect.size;

    double width = size.width;
    double height = size.height;
    if (width < height)
        swap(width, height);

    double imgSq = imgSize.area();
    double rectSq = rect.size.area();
    
    // Отношение площади контура к его Minimum Rectangle
    if (imgSq / rectSq < 4)
        return false;

    // Оношенеи сторон бокса
    if (width > height * 6)
        return false;

    return true;
}

// Проверка
bool sizeCheckPlate(vector<Point>& convex) {
    double width = 0.;
    double height = 0.;

    // Нахождение средней высоты и ширины бокса 
    for (int i = 0; i < 4; ++i) {
        Point v = convex[i] - convex[(i + 1) % 4];

        if (i % 2)
            width += sqrt(v.ddot(v));
        else
            height += sqrt(v.ddot(v));
    }
    width /= 2;
    height /= 2;
    
    if (width < height)
        swap(width, height);

    // Проверка отношения сторон номера
    if (width > height * 6 || width < height * 3)
        return false;

    return true;
}

// Вырезание части изображения по четырем точкам
void getSubImage(Mat &img, vector<Point>& convex, vector<Mat> &potentialPlate) {
    constexpr float width = 520;
    constexpr float height = 112;
    vector<Point2f> convex2f;

    vector<Point2f> dst = { {0, 0}, {width, 0}, {width, height}, {0, height} };

    int longestIndx = 0;
    double length2 = 0;
    for (int i = 0; i < 4; ++i) {
        Point v = convex[i] - convex[(i + 1) % 4];
        if (v.ddot(v) > length2) {
            length2 = v.ddot(v);
            longestIndx = i;
        }
    }

    if (convex[longestIndx].y > convex[(longestIndx + 3) % 4].y)
        longestIndx = (longestIndx + 2) % 4;

    for (int i = 0; i < 4; ++i) {
        convex2f.push_back(convex[(longestIndx + i) % 4]);
    }
    
    Mat M = getPerspectiveTransform(convex2f, dst), cropped;

    warpPerspective(img, cropped, M, Size(width, height));
    potentialPlate.push_back(cropped);
}

// N = p0 + t * (p1 - p0)
// N = p2 + s * (p3 - p2)

// 0 = (p0 - p2) + t * (p1 - p0) - s * (p3 - p2)
// s * (p3 - p2) - t * (p1 - p0) = (p0 - p2)

// (u.x v.x) * (s)  = (p.x)
// (u.y v.y) * (-t) = (p.y)

// (s)  = (v.y -v.x) * (p.x)
// (-t) = (-u.y u.x) * (p.y)

bool findIntersect(Point p0, Point p1, Point p2, Point p3, Point &res) {
    Point v = p1 - p0;
    Point u = p3 - p2;
    Point p = p0 - p2;

    double D = v.y * u.x - v.x * u.y;
    
    if (D == 0)
        return false;

    double s = (v.y * p.x - v.x * p.y) / D;

    res = p2 + u * s;
    return true;
}

double findAdditionalSquare(Point p0, Point p1, Point p2, Point p3) {
    Point point;
    bool status = findIntersect(p0, p1, p2, p3, point);
    
    if (!status)
        return 1e14;

    return abs((p2 - point).cross(p1 - point));
}

void postProccess(vector<Point>& points) {
    while (points.size() != 4) {
        int erraseIndx = 0;
        double minSq = 1e14;
        size_t size = points.size();

        for (int i = 0; i < size; ++i) {
            double sq = findAdditionalSquare(points[i], points[(i + 1) % size], points[(i + 2) % size], points[(i + 3) % size]);

            if (sq < minSq) {
                minSq = sq;
                erraseIndx = i;
            }
        }

        Point point;
        findIntersect(points[erraseIndx], points[(erraseIndx + 1) % size], points[(erraseIndx + 2) % size], points[(erraseIndx + 3) % size], point);
        points[(erraseIndx + 1) % size] = point;
        points.erase(points.begin() + ((erraseIndx + 2) % size));
    }
}

struct RectSort {
    bool operator()(const Rect& a, const Rect& b) const{
        return a.tl().x < b.tl().x;
    }
};

// Отбор контуров, подходящих под цифру
vector<Rect> cleanUpContours(vector<vector<Point>> &contours) {
    vector<Rect> rects;
    vector<Rect> filteredRects;

    for (int i = 0; i < contours.size(); ++i) {
        auto& c = contours[i];
        auto rect = boundingRect(c);

        // Фильтрация контуров по размерам
        if (rect.width < 60 && rect.width > 25 && rect.height < 100 && rect.height > 40)
            rects.push_back(rect);
    }

    // Упорядоочивание контуров
    sort(rects.begin(), rects.end(), RectSort());

    if (rects.size() > 0) {
        filteredRects.push_back(rects[0]);
        for (int i = 1; i < rects.size(); ++i) {
            if (rects[i - 1].br().x < rects[i].br().x)
                filteredRects.push_back(rects[i]);
        }
    }

    return filteredRects;
}

string findMinScore(Mat &image, vector<pair<Mat, string>> &masks) {
    auto element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(image, image, MORPH_ERODE, element);
    int minScore = 1'000'000;
    string value;

    for (auto& lp : masks) {
        Mat res;
        bitwise_xor(image, lp.first, res);
        int score = countNonZero(res);

        if (score < minScore) {
            minScore = score;
            value = lp.second;
        }
    }

    return value;
}

string recognizeSymbols(Mat &plate, vector<Rect> &rects, vector<pair<Mat, string>> &digitsMask, vector<pair<Mat, string>>& lattersMask) {
    string plateValue;

    auto symbol = plate(rects[0]);
    resize(symbol, symbol, lattersMask[0].first.size(), 0, 0, INTER_NEAREST);
    plateValue.append(findMinScore(symbol, lattersMask));

    for (int i = 1; i < 4; ++i) {
        symbol = plate(rects[i]);
        resize(symbol, symbol, digitsMask[0].first.size());
        plateValue.append(findMinScore(symbol, digitsMask));
    }

    for (int i = 4; i < 6; ++i) {
        symbol = plate(rects[i]);
        resize(symbol, symbol, lattersMask[0].first.size());
        plateValue.append(findMinScore(symbol, lattersMask));
    }

    for (int i = 6; i < rects.size(); ++i) {
        symbol = plate(rects[i]);
        resize(symbol, symbol, digitsMask[0].first.size());
        plateValue.append(findMinScore(symbol, digitsMask));
    }

    return plateValue;
}

vector<pair<Mat, string>> loadMasks(fs::path path) {
    vector<pair<Mat, string>> masks;

    for (const auto& dirEntry : fs::directory_iterator(path))
        if (fs::is_regular_file(dirEntry)) {
            auto filename = dirEntry.path().filename().string();
            size_t lastindex = filename.find_last_of(".");
            filename = filename.substr(0, lastindex);
            masks.push_back({ imread(dirEntry.path().string(), IMREAD_GRAYSCALE), filename });
        }

    return masks;
}

struct ImageData {
    string value;
    vector<Point> plate;
};

map<string, ImageData> loadData(fs::path path) {
    map<string, ImageData> data;

    ifstream in;
    in.open(path.string());

    while (!in.eof()) {
        ImageData d;
        string name;
        
        in >> name;
        for (int i = 0; i < 4; ++i) {
            int x, y;
            in >> x >> y;

            d.plate.push_back(Point(x, y));
        }
        
        in >> d.value;
        data[name] = d;
    }

    return data;
}

double IoUScore(Size size, vector<Point> &p1, vector<Point> &p2) {
    Mat un(size, CV_8UC1), intersect(size, CV_8UC1), p1Im(size, CV_8UC1), p2Im(size, CV_8UC1);
    vector<vector<Point>> _p1, _p2;

    _p1.push_back(p1);
    _p2.push_back(p2);
    
    un.setTo(Scalar(0));
    p1Im.setTo(Scalar(0));
    p1Im.setTo(Scalar(0));

    if (!p1.empty()) {
        drawContours(un, _p1, 0, Scalar(255), -1);
        drawContours(p1Im, _p1, 0, Scalar(255), -1);
    }

    if (!p2.empty()) {
        drawContours(un, _p2, 0, Scalar(255), -1);
        drawContours(p2Im, _p2, 0, Scalar(255), -1);
    }


    bitwise_and(p1Im, p2Im, intersect);

    double unCount = countNonZero(un);
    double intersectCount = countNonZero(intersect);

    return intersectCount / unCount;
}

enum ConfMat {
    TP = 0,
    FP,
    TN,
    FN
};

struct score {
    double iou;
    ConfMat type;

    bool operator<(const score& a) {
        return iou < a.iou;
    }
};

double calMAP(vector<score>& scores) {
    vector<pair<double, double>> pr;
    pr.push_back({ 1., 0. });

    int tp = 0;
    int fp = 0;

    for (auto& s : scores) {
        if (s.type == FP)
            fp++;
        if (s.type == TP)
            tp++;

        if (tp == 0 && fp == 0)
            pr.push_back({ 1., 0 });
        else
            pr.push_back({ (double) tp / (tp + fp), (double)tp / scores.size()});
        std::cout << s.iou << std::endl;
    }

    auto start = pr[0];
    double res = 0;

    for (auto p : pr) {
        std::cout << p.first << " " << p.second << std::endl;
        res += ((start.first + p.first) / 2) * (p.second - start.second);
        start = p;
    }

    return res;
}

int main(int argc, const char** argv)
{
    auto path = filesystem::current_path().append("carPlate");

    auto digitsMask = loadMasks(path / fs::path("digitsMask"));
    auto lattersMask = loadMasks(path / fs::path("lattersMask"));

    map<string, ImageData> data = loadData(path / "data" / "data.txt");

    vector<score> scores;

    for (const auto& dirEntry : fs::directory_iterator(path)) {
        if (fs::is_regular_file(dirEntry)) {
            auto start = chrono::steady_clock::now();
            Mat image = imread(dirEntry.path().string(), IMREAD_GRAYSCALE), copy, out;
            auto filename = dirEntry.path().filename().string();
            size_t lastindex = filename.find_last_of(".");
            filename = filename.substr(0, lastindex);
            copy = image.clone();

            // Бинаризация
            threshold(copy, copy, 130, 255, THRESH_BINARY);

            Mat mask(copy.size(), CV_8UC1);
            mask.setTo(Scalar(0));

            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;

            // Поиск контуров
            findContours(copy, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

            // Предотбор контуров, формирование маски номера
            for (int j = 0; j < contours.size(); ++j){
                auto rect = minAreaRect(contours[j]);
                auto contourSq = contourArea(contours[j]);
                auto rectSq = rect.size.area();

                // Удаление контуров с маленькой площадью
                // Удаление контуров с сильно большой площадью
                // Удаление контуров не похожих на прямоугольник (проверка отношения площадей)
                if (rectSq / contourSq < 2 && contourSq > 700 && sizePreCheck(mask.size(), rect)) {
                    drawContours(mask, contours, j, Scalar(255), -1);
                    drawContours(mask, contours, j, Scalar(255), 5);
                }
            }

            // Пост обработка маски, поиск номерной рамки
            contours.clear();
            findContours(mask, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

            mask.setTo(Scalar(0));
            vector<Mat> potentialPlate;
            vector<vector<Point>> potentialPlateContours;

            // Перебор контуров, потенциальных для номерной рамки
            for (int num = 0; num < contours.size(); ++num) {
                // Выпуклая оболочка контура
                vector<Point> convex;
                convexHull(contours[num], convex);

                // Спрямление выпуклой оболочки до четырех точек
                postProccess(convex);

                // Более строгая проверка размера номерной рамки и вырезание соотвтствующей части изображения
                if (sizeCheckPlate(convex)) {
                    getSubImage(image, convex, potentialPlate);
                    potentialPlateContours.push_back(convex);
                }
            }

            string plateValue;
            vector<Point> plate;
            // Обработка потенциальных номерных рамок
            for (int num = 0; num < potentialPlate.size(); ++num) {
                // Адаптивная бинаризация
                adaptiveThreshold(potentialPlate[num], potentialPlate[num], 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 13, 3);

                // Морфологические операии для удаления мелкого, контрастного шума с изображения
                auto element = getStructuringElement(MORPH_RECT, Size(3, 3));
                morphologyEx(potentialPlate[num], potentialPlate[num], MORPH_DILATE, element);
                morphologyEx(potentialPlate[num], potentialPlate[num], MORPH_ERODE, element);

                // Поиск контуров цифр и букв
                vector<vector<Point>> symbolContours;
                findContours(potentialPlate[num], symbolContours, RETR_TREE, CHAIN_APPROX_NONE);
                
                // Чистка набора контуров, удаление не подходящих контуров
                auto rects = cleanUpContours(symbolContours);
                
                if (rects.size() > 6 && rects.size() < 10) {
                    plateValue = recognizeSymbols(potentialPlate[num], rects, digitsMask, lattersMask);
                    plate = potentialPlateContours[num];
                    imshow(dirEntry.path().filename().string(), potentialPlate[num]);
                }
            }
            auto end = chrono::steady_clock::now();
            cout << "Time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;
            if (data.find(filename) != data.end()) {
                for (auto &p : potentialPlateContours) {
                    double iou = IoUScore(image.size(), data[filename].plate, p);
                    scores.push_back({ iou, (iou < 0.7 ? (plate.empty() ? FN : FP) : TP)});
                }
                // cout << "IoU score: " << iou << endl;
                // Вывод информации о распозновании
                if (plateValue != data[filename].value)
                    cout << "ERROR! For file: " << filename << ", expected: " << data[filename].value << ", got: " << plateValue << endl;
                else
                    cout << "OK! For file: " << filename << ", license plate: " << plateValue << endl;
            }
            else
                if (plateValue.empty())
                    cout << "Number not found!" << endl;
                else
                    cout << "Number is: " << plateValue << endl;
            cout << endl;
        }
    }

    sort(scores.rbegin(), scores.rend());

    std::cout << calMAP(scores) << std::endl;

    cv::waitKey();

    return 0;
}