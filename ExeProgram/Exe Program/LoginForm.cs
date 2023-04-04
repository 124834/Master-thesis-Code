
using BLL;
using Model;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;

using System.Windows.Forms;
using TemperatureClient;



namespace insulator
{
    public partial class LoginForm : Form
    {

        private int count = 0;
        public LoginForm()
        {
            InitializeComponent();   //主要是用来初始化designer时拖到Form上的Control的。比如说你拖上取一个TextBox,他放在Form的什么位置拉，TextBox的一些属性拉。包括new 这个TextBox都放在那个函数里面处理的。
        }



        private void LoginForm_Load(object sender, EventArgs e)
        {
           
            this.timer1.Enabled = true;   //创建定时器
            this.timer1.Interval = 1000;    //10ms产生一次中断
        }

        private void LoginForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            Application.Exit(); //关闭控件
        }

        private void timer1_Tick(object sender, EventArgs e)   //定时器中断
        {
            count++;
            this.label1.Text = "time:" + (5 - count);
            if (count > 4)
            {
                SmartCupSystem monitor = new SmartCupSystem();
                monitor.Show();
                this.timer1.Enabled = false;
                this.count = 0;
                this.Hide();
            }
        }

        private void button1_Click(object sender, EventArgs e)
        {
            
            this.timer1.Enabled = false;
            this.count = 0;
            this.Hide();
        }

        //按键按下调用函数
        private void button2_Click(object sender, EventArgs e)
        {
            SmartCupSystem monitor = new SmartCupSystem();   //分配内存
            monitor.Show();                    //调用显示控件
            this.timer1.Enabled = false;       //关闭定时器
            this.count = 0;                    //变量赋值
            this.Hide();                       //$(this).hide()是jQuery里面的隐藏功能，表示隐藏当前元素，确切点应该指的是对象，比如有一个按钮，你对该按钮添加这个点击事件就可以点击此按钮，实现隐藏这个按钮的功能，这里的this表示当前对象，就是你鼠标点击的那个页面元素对象
        }

        private void label3_Click(object sender, EventArgs e)
        {

        }

    }
}
