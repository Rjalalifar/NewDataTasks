# NewDataTasks

1-Routing:
To use this Routing code you can use Spyder or any IDEs to run my code. This code is reading routing api utilizing OSRM-Backend that has been impelemented in our server by the address of "http://62.106.95.167:5000/route/v1/driving/" to route.
You can see an example of routing in tehran in bellow picture
![alt text](https://github.com/Rjalalifar/NewDataTasks/blob/main/Images/RoutingResult.png)

codes are comented and you can understand each step by comments.

2-PassageSummery:
In this codes you can insert your full text in "persian_text" variable.
If your Text file in in persian decomment "Language_Variable="Persian"" . by default it summerize english passage.
Do to the fact that hazm site-package had some problems in installation and ijust need persian stopword from hazm; i did not use it and i created persian_stopwords array at the begining of the codes to use it for summerization.
codes are commented and u can understand each step from comments.

![alt text](https://github.com/Rjalalifar/NewDataTasks/blob/main/Images/SummaryResult.png)

3-Trraffic Forecasting: This code is to modeling and predicting traffic. It predicts speed of drivindg in different roads. the original model than i costomized to my project is created by Yu, Bing, Haoteng Yin, and Zhanxing Zhu. . you can find it in :https://github.com/VeritasYin/STGCN_IJCAI-18
Firstly data is downloading from osm by osmnx framework and then model runs. you can use google colab of any IDEs such as Spyder to run it.
![alt text](https://github.com/Rjalalifar/NewDataTasks/blob/main/Images/TrafficData.png)
![alt text](https://github.com/Rjalalifar/NewDataTasks/blob/main/Images/TrafficResult.png)
