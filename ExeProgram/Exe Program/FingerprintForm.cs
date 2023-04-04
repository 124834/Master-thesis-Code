using Model;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace LampControl
{
    public partial class FingerprintForm : Form
    {
        private int fNumber = 0;
        private TemperatureClient.SmartCupSystem monitorForm;
        /// <summary>
        /// 文件路径
        /// </summary>
        private string dataPath = System.Windows.Forms.Application.StartupPath + "\\perData.log";
        /// <summary>
        /// 学生集合
        /// </summary>
        List<StudentModel> stus = new List<StudentModel>();
        private static object logWriterLock = new object();
        public FingerprintForm()
        {
            InitializeComponent();
            //禁止线程间调用控件校验
            Control.CheckForIllegalCrossThreadCalls = false;
        }

        public FingerprintForm(TemperatureClient.SmartCupSystem monitorForm)
        {
            InitializeComponent();
            //禁止线程间调用控件校验
            Control.CheckForIllegalCrossThreadCalls = false;
            this.monitorForm = monitorForm;
        }

        private void FingerprintForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            e.Cancel = true;
            this.Visible = false;
        }

        private void FingerprintForm_Load(object sender, EventArgs e)
        {
            show();
            

        }

        private void show()
        {
            try
            {
                this.dataGridViewData.Rows.Clear();
            }
            catch (Exception)
            {
            }

            try
            {
                this.dataGridViewData.Columns.Clear();
            }
            catch (Exception)
            {

            }
            this.stus = getStudentDatas();
            this.dataGridViewData.DataSource = stus;
            DataGridViewButtonColumn btn = new DataGridViewButtonColumn();
            btn.Name = "colbtn";
            btn.HeaderText = "删除";
            btn.DefaultCellStyle.NullValue = "删除";
            dataGridViewData.Columns.Add(btn);
            this.textBoxFingerprint.Text = (this.fNumber + 1).ToString();
            this.textBoxFingerprint.Enabled = false;

            this.dataGridViewData.Columns[3].Visible = false;

        }


        /// <summary>
        /// 获取所有数据
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        public List<StudentModel> getStudentDatas()
        {
            List<StudentModel> sms = new List<StudentModel>();
            try
            {
                lock (logWriterLock)
                {
                    StreamReader sr = new StreamReader(this.dataPath, Encoding.UTF8);
                    try
                    {
                        String line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            try
                            {
                                StudentModel sm = new StudentModel();
                                string[] strs = line.Split('&');
                                if (strs.Length == 4)
                                {
                                    sm.Number = strs[0].Trim();
                                    int tempNumber=Convert.ToInt32(sm.Number);
                                    if (tempNumber > this.fNumber)
                                    {
                                        this.fNumber = tempNumber;
                                    }
                                    sm.Name = strs[1].Trim();
                                    sm.Department = strs[2].Trim();
                                    
                                    sms.Add(sm);
                                }

                            }
                            catch (Exception)
                            {

                            }
                        }
                        sr.Close();
                    }
                    catch (Exception)
                    {

                    }
                    finally
                    {
                        sr.Close();
                    }
                }
                return sms;
            }
            catch (Exception)
            {
                return sms;
            }
            // return ds;
        }

        private void dataGridViewData_CellContentClick(object sender, DataGridViewCellEventArgs e)
        {
            if (this.dataGridViewData.Columns[e.ColumnIndex].Name == "colbtn")
            {
                string curNumber = this.dataGridViewData.Rows[e.RowIndex].Cells[0].Value.ToString();
                FileStream fs2 = new FileStream(this.dataPath, FileMode.Truncate, FileAccess.ReadWrite);
                StreamWriter sw = new StreamWriter(fs2, Encoding.UTF8); 
                foreach (StudentModel item in stus)
                {
                    if (!item.Number.Equals(curNumber))
                    {
                        string curStr = item.Number + "&" + item.Name + "&" + item.Department + "&"  + item.IsDel;
                        sw.WriteLine(curStr);
                    }
                }
                sw.Close();
                fs2.Close();


               
                show();
            }
        }

        /// <summary>
        /// 写文件
        /// </summary>
        /// <param name="message"></param>
        private void WriteValue(string message)
        {
            lock (logWriterLock)
            {
                StreamWriter logWriter2 = new StreamWriter(this.dataPath, true);
                try
                {
                    logWriter2.WriteLine(message.Trim());
                    logWriter2.Close();
                }
                catch (Exception)
                {
                }
                finally
                {
                    logWriter2.Close();
                }
            }
        }


        /// <summary>
        /// 添加
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonAdd_Click(object sender, EventArgs e)
        {
            string number = this.textBoxFingerprint.Text.Trim();
            if (String.IsNullOrEmpty(number))
            {
                MessageBox.Show("指纹ID不能为空！");
                return;
            }
           


            string name = this.textBoxName.Text.Trim();
            if (String.IsNullOrEmpty(name))
            {
                MessageBox.Show("姓名不能为空！");
                return;
            }
            if (name.Length > 15)
            {
                MessageBox.Show("姓名长度不能超过15，请重新输入");
                this.textBoxName.Focus();
                return;
            }

            foreach (StudentModel item in stus)
            {
                if (item.Number.Equals(number))
                {
                    MessageBox.Show("指纹ID已经存在！");
                    return;
                }
                if (item.Name.Equals(name))
                {
                    MessageBox.Show("姓名已经存在！");
                    return;
                }
            }
            string classs = this.textBoxClass.Text.Trim();
            if (String.IsNullOrEmpty(classs))
            {
                MessageBox.Show("部门不能为空！");
                return;
            }
           
            string stuStr = number + "&" + name + "&" + classs +"&0";
            WriteValue(stuStr);
            show();
            this.textBoxClass.Text = "";
            this.textBoxName.Text = "";

        }

        /// <summary>
        /// 输入指纹按钮点击事件
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void buttonFinger_Click(object sender, EventArgs e)
        {
            byte[] bytes=new  byte[1];
            
            bytes[0] = System.BitConverter.GetBytes(Convert.ToInt32(this.textBoxFingerprint.Text))[0];
            this.monitorForm.sendData(bytes);
        }



    }
}
