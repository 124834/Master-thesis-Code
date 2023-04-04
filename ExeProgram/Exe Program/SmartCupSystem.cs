using BLL;
using LampControl;
using Model;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Configuration;
using System.Data;
using System.Drawing;
using System.IO;
using System.IO.Ports;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace TemperatureClient
{
    public partial class SmartCupSystem : Form
    {
        /// <summary>
        
        /// </summary>
        private byte[] byData = new byte[2];

        /// <summary>
        
        /// </summary>
        private int showCount = 0;

        /// <summary>
      
        /// </summary>
        private List<DataModel> ds;

        /// <summary>
     
        /// </summary>
        private static object logWriterLock = new object();

        private FingerprintForm fpForm;

        byte[] bs = new byte[3];

        private int mianji = 49;
        private int total_water = 440;
        private int water_flag = 0;
        private int volume_flag = 0;
        private int volume = 0;
        private string dStr = " ";
        private string str_start_time = " ";
        private string str_stop_time = " ";
        private int water_start = 0;
        private int time_flag = 0; 
        private string[] str_name = new string[]{"  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "
                                     ,"   ", "   ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "
                                     ,"   ", "   ", "  ", "  ", "  ", "  ", "  ", "  ", "  ", "  "};


        private static string dataPath_name = System.Windows.Forms.Application.StartupPath + "\\SaveData.txt";
        private static string dataPath_xls = System.Windows.Forms.Application.StartupPath + "\\SaveData.xls";

        private static void Write_name(string message)
        {
            lock (logWriterLock)
            { StreamWriter logWriter2 = new StreamWriter(dataPath_name, true);
                try
                {  logWriter2.WriteLine(message.Trim());
                    logWriter2.Close(); }
                catch (Exception)
                { }
                finally
                {logWriter2.Close();}
            }
        }

        private static void Write_name_xls(string message)
        { lock (logWriterLock)
            {   StreamWriter logWriter2 = new StreamWriter(dataPath_xls, true);
                try
                {  logWriter2.WriteLine(message.Trim());
                    logWriter2.Close();}
                catch (Exception)
                {}
                finally
                { logWriter2.Close(); }
            }
        }
        public SmartCupSystem()
        { InitializeComponent();          
            
            Control.CheckForIllegalCrossThreadCalls = false; }

        private void Monitor_Load(object sender, EventArgs e)
        {
            fpForm = new FingerprintForm(this);
            fpForm.Show();
            fpForm.Visible = false;
           // this.fpForm.textBoxFingerprint.Text = "";
            setComs();
            this.textBox7.Text = "440";
            this.textBox6.Text = "49";
            this.timer1.Enabled = true;
            this.timer1.Start();

        }


        /// <summary>
        /// 
        /// </summary>
        private void setComs()
        { this.comboBoxRecComs.Items.Clear();
            string[] coms = SerialPort.GetPortNames();
            foreach (string comName in coms)
            {  this.comboBoxRecComs.Items.Add(comName);}
            try
            {  this.comboBoxRecComs.SelectedIndex = 0;}
            catch (Exception)
            {}
        }

        private void buttonComRefresh_Click(object sender, EventArgs e)
        {  setComs(); }

        private void buttonRecOpenCom_Click(object sender, EventArgs e)
        { if ("关闭".Equals(this.buttonRecOpenCom.Text.Trim()))
            {  this.serialPort1.Close();
                this.buttonComRefresh.Enabled = true;
            }
            else
            { try
                {   serialPort1.PortName = this.comboBoxRecComs.Text;
                    this.serialPort1.Open();
                    if (serialPort1.IsOpen == true)
                    {  this.buttonComRefresh.Enabled = false; }
                    else
                    {  MessageBox.Show("Serial port opening failed！");
                        return; }
                }
                catch (Exception ex)
                { MessageBox.Show("Serial port opening failed！");
                    return;}
            }
            this.buttonRecOpenCom.Text = "open".Equals(this.buttonRecOpenCom.Text.Trim()) ? "close" : "open";
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="bytes"></param>
        /// <returns></returns>
        private string bytesToHexString(byte[] bytes)
        {  string hexString = string.Empty;
            if (bytes != null)
            {
                StringBuilder strB = new StringBuilder();
                for (int i = 0; i < bytes.Length; i++)
                { strB.Append(bytes[i].ToString("X2"));}
                hexString = strB.ToString();
            } return hexString;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void serialPort1_DataReceived(object sender, SerialDataReceivedEventArgs e)
        {  try
            {  Thread.Sleep(50);  
                byte[] bt = new byte[serialPort1.BytesToRead];
                serialPort1.Read(bt, 0, serialPort1.BytesToRead);
                string reStr = System.Text.Encoding.ASCII.GetString(bt).Trim();
                
                string nStr = this.bytesToHexString(bt);
                string a;
                a = Convert.ToString(bt[3], 10);
                int jiaodu_x = int.Parse(a);

                a = Convert.ToString(bt[4], 10);
                int jiaodu_y = int.Parse(a);

                a = Convert.ToString(bt[5], 10);
                int juli_1 = int.Parse(a);

                a = Convert.ToString(bt[6], 10);
                int juli_2 = int.Parse(a);

                int juli = 93-juli_1 * 100 - juli_2; // water height
                if((jiaodu_x<20)&& (jiaodu_y < 20))
                {
                    volume = mianji * juli/10;
                }
                
                this.textBox2.Text = jiaodu_x + "°/" + jiaodu_y + "°";
                this.textBox3.Text = volume + "mL";
                this.textBox4.Text = juli + "mm" + "°/"+ water_start+ "°/"+volume_flag;
                
              

                if ((jiaodu_x > 20) || (jiaodu_y > 20))
                {
                    if (water_start == 0)
                    {  str_start_time = dStr;  //
                        water_start = 1; //
                        volume_flag = volume;
                        time_flag = 2; }
                }
                else {
                    
                    if ((water_start == 1)&&(time_flag==0))
                    { if ((jiaodu_x < 20) && (jiaodu_y < 20))
                        {  str_stop_time = dStr;  
                            water_start = 0; 
                            time_flag = 2;
                            //MessageBox.Show("Success");
                        }
                    }
                }
                
            }
            catch (Exception)
            { //this.labelMsg.ForeColor = System.Drawing.Color.Red;
                //this.labelMsg.Text = "Wrong！";
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="cmdType"></param>
        /// <param name="byteData"></param>
        /// <returns></returns>
        private bool sendData()
        {
            if ("关闭".Equals(this.buttonRecOpenCom.Text.Trim()))
            {
                this.serialPort1.Write(byData, 0, byData.Length);//
                return true;
            }
            return false;
        }

        /// <summary>
        ///
        /// </summary>
        /// <param name="cmdType"></param>
        /// <param name="byteData"></param>
        /// <returns></returns>
        public bool sendData(Byte[] bytes)
        {
            if ("关闭".Equals(this.buttonRecOpenCom.Text.Trim()))
            {
                this.serialPort1.Write(bytes, 0, bytes.Length);//发送
                return true;
            }
            return false;
        }

        private void Monitor_FormClosed(object sender, FormClosedEventArgs e)
        {
            this.fpForm.Visible = false;
            this.fpForm.Close();
            try
            {
                this.fpForm.Dispose();
            }
            catch (Exception)
            {

            }
            Application.Exit();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            
            //zhiwen_num = int.Parse(comboBox1.Text);

           
            string num_1 = this.textBox6.Text.Trim();
            string num_2 = this.textBox7.Text.Trim();

            
            if ((num_1 == "")||(num_2 == "")){
                MessageBox.Show("录入信息不能为空");
            }
            else { 
                //
                //Write_name(comboBox1.Text + "&" + remarks);   
                //MessageBox.Show("Success");
                mianji =  int.Parse(num_1);
                total_water = int.Parse(num_2);
            }
            //this.label1.Text = comboBox1.Text + "&&" + remarks;
        }

        private void button2_Click(object sender, EventArgs e)
        {
            string filePath = dataPath_name;
            this.listBox1.Items.Clear();
            int lines = 0;
            int i = 0;

            if (File.Exists(filePath))
            {
                using (StreamReader sr = new StreamReader(filePath, Encoding.UTF8))
                {
                    //label6.Text = sr.ReadToEnd();
                    //this.listBox1.Text = sr.ReadToEnd();
                    //this.checkedListBox1.Text = sr.ReadToEnd();
                    
                    while (sr.ReadLine() != null)
                    {
                        lines++;      
                       
                    }
                    //card_total = lines;
                }

                using (StreamReader sb = new StreamReader(filePath, Encoding.UTF8))
                {
                    for (i = 0; i < lines; i++)
                    {
                        //this.listBox1.Items.Add(sb.ReadLine());   
                        

                        string st = sb.ReadLine();  

                        string[] sArray_2 = st.Split('&');  
                        this.listBox1.Items.Add(sArray_2[0]);                    
                    }
                }
            }
            else
            {
                MessageBox.Show("No access");
            }
        }

        
        private byte[] StringToBytes(string TheString)
        {
            Encoding FromEcoding = Encoding.GetEncoding("UTF-8");
            Encoding ToEcoding = Encoding.GetEncoding("GB2312");
            byte[] FromBytes = FromEcoding.GetBytes(TheString);
            byte[] ToBytes = Encoding.Convert(FromEcoding, ToEcoding, FromBytes);
            return ToBytes;
        }

        /*
        private void button4_Click(object sender, EventArgs e)
        {
            string save_data = this.textBox4.Text.Trim();
            
            if (save_data == "")
            {
                MessageBox.Show("录入信息不能为空");
            }
            else
            {
                byte[] StringToByte = StringToBytes(textBox4.Text);    
                byte[] bs_a = new byte[1];
                bs_a[0] = 0xfd;
                sendData(bs_a);
                sendData(StringToByte);
                bs_a[0] = 0xff;
                sendData(bs_a);
                MessageBox.Show("Success");
            }
        }
        */

        private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
        {

        }

        
        private void timer1_Tick_1(object sender, EventArgs e)
        {
            dStr = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            this.textBox1.Text = dStr;
            if (time_flag > 0)
            {
                time_flag--;
                if (time_flag==0)
                {
                                     
                    if (volume_flag > volume)
                    {
                        water_flag = volume_flag - volume;
                    }
                    else
                    {
                        water_flag = 0;
                    }
                    this.textBox5.Text = water_flag + "mL";
                    

                    Write_name("start:" + str_start_time + "  stop:" + str_stop_time + "  Water Intake:" + water_flag + "mL" + "&");   //写入文件
                    Write_name_xls("start:" + str_start_time + "  stop:" + str_stop_time + "  Water Intake:" + water_flag + "mL" + "&");
                }
            }
        }

        private void label4_Click(object sender, EventArgs e)
        {

        }

        private void textBox6_TextChanged(object sender, EventArgs e)
        {

        }

        private void textBox4_TextChanged(object sender, EventArgs e)
        {

        }
    }
}


