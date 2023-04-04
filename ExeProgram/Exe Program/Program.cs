
using insulator;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;

namespace TemperatureClient
{
    static class Program
    {
        /// <summary>
        /// 应用程序的主入口点。
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();//的作用是激活应用程序的显示风格，而显示风格是构成操作系统主题的各种元素，如色彩、字体等。如果控件和OS支持，那么控件的绘制就会根据显示风格来实现。

            Application.SetCompatibleTextRenderingDefault(false);//某些窗体控件在给它们的文本着色时可以使用 TextRenderer类也可以使用 Graphics类。
            // Application.Run(new Monitor());
             Application.Run(new LoginForm());
            //Application.Run(new ChartMonitor());
        }
    }
}


