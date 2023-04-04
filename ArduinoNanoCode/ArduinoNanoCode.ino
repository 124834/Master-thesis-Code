#include "I2Cdev.h"
#include "MPU6050.h"
#include <Arduino.h>
//#include <U8g2lib.h>

#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>
#endif
#ifdef U8X8_HAVE_HW_I2C
#include <Wire.h>
#endif
#include <SD.h>
#define CS 4
File logFile;

//U8G2_SSD1306_128X64_NONAME_F_SW_I2C u8g2(U8G2_R0, /* clock=*/ A1, /* data=*/ A0, /* reset=*/ U8X8_PIN_NONE);   // 设置D10做SCL，D11做SDA，可以改为其它接口

MPU6050 accelgyro;

unsigned int ax, ay, az;
unsigned int  gx, gy, gz;
char str_a[50];
int juli = 0;
//超声波引脚定义
int TrgPin = 3;
int EcoPin = 2;
int i = 0;
int num = 0;
float time=0;
int volume=0;
void read_juli()
{
    digitalWrite(TrgPin, LOW);
    delayMicroseconds(8);
    digitalWrite(TrgPin, HIGH);
    // 维持10毫秒高电平用来产生一个脉冲
    delayMicroseconds(10);
    digitalWrite(TrgPin, LOW);
    // 读取脉冲的宽度并换算成距离
    juli = pulseIn(EcoPin, HIGH)*10 / 58;   //转换成真实数值单位mm
    if(juli>999)
      juli = 999;
}

void switch_mpu6050()
{
      if(ax>30000)
      {
        ax =65536- ax; 
      }
      ax = ax/175;
      if(ay>30000)
      {
          ay = 65536-ay; 
      }
      ay = ay/175;  

       if(az>30000)
      {
          az = 65536-az; 
      }
      az = az/175; 

         if(gx>30000)
      {
      gx = 65536-gx; 
      }
      gx = gx/175; 

      if(gy>30000)
      {
          gy = 65536-gy; 
      }
      gy = gy/175; 
      
      if(gz>30000)
      {
          gz = 65536-gz; 
      }
      gz = gz/175; 
  
}

void send_data()
{
    Serial.write(0xaa);
    Serial.write(0xFB);
    Serial.write(0xFF);
    Serial.write(0xFF);
    Serial.write(ax); //角度
    Serial.write(ay); //角度
    Serial.write(juli/100);
    Serial.write(juli%100);
}

//此函数的Serial.println为打印信息，调试完成之后，用//屏蔽掉，否则影响上传
boolean initCard()
{
  // Serial.print("Connecting to SD card... ");

  // Initialize the SD card
  if (!SD.begin(CS))
  {
    // An error occurred
    // Serial.println("\nError: Could not connect to SD card!");
    // Serial.println("- Is a card inserted?");
    // Serial.println("- Is the card formatted as FAT16/FAT32?");
    // Serial.println("- Is the CS pin correctly set?");
    // Serial.println("- Is the wiring correct?");

    return false;
  }
  else
    //Serial.println("Done!");

  return true;
}

void setup (void) 
{
    delay(200); 
    
    Serial.begin(9600); //初始化串口 

    //此部分代码初始化SD卡
    if (!initCard())
      while (1);
    //如果打开失败，可以在sd卡的根目录下创建log.txt文件  
    logFile = SD.open("log.txt", FILE_WRITE);

    if (!logFile)
    {
      //调试成功，屏蔽此行代码
      //Serial.println("Could not open logfile!");
      while (1);
    }
     else
    //Serial.println("File Done!");
  
    accelgyro.initialize();  //初始化MPU6050

    // 设置TrgPin为输出状态
    pinMode(TrgPin, OUTPUT);delay(10);
    // 设置EcoPin为输入状态
    pinMode(EcoPin, INPUT);delay(10);  
    
    //u8g2.begin();
    delay(10);
}

//accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
void loop (void)
{
   delay(200);    //延时0.2s 防止程序执行过快
   i++;
   accelgyro.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);  //读取角度 ax是角度 gx是加速度
   //mpu6050转换
   switch_mpu6050();

   read_juli(); //读取距离
   volume=(98-juli)*4902/1000;
   send_data(); //发送数据
   
   //保存有效数据到数组中
   if(i%1==0)
   {  
       num++;
       time=num*0.1;//单位是s
      logFile = SD.open("log.txt", FILE_WRITE);
       if(logFile)
       {
           //logFile.println(str_a);
           logFile.print("Time:");logFile.print(time); logFile.print("s  ");
           logFile.print("ax:");logFile.print(ax); logFile.print("   ");
           logFile.print("ay:");logFile.print(ay); logFile.print("   ");
           logFile.print("az:");logFile.print(az); logFile.print("   ");
           logFile.print("gx:");logFile.print(gx); logFile.print("   ");
           logFile.print("gy:");logFile.print(gy); logFile.print("   ");
           logFile.print("gz:");logFile.print(gz); logFile.print("   ");
           logFile.print("Remaining Water/ml:");logFile.println(volume);
           logFile.close();
          //Serial.println("单次完成!");Serial.println(juli);
       }
        else
        {//Serial.println("写入失败!");
        }
       
    }   
}
  