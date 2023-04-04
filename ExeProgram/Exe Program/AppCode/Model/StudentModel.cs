using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace Model
{
     [Serializable]
    public class StudentModel
    {
        public string Number { get; set; }

        public string  Name { get; set; }

        public string Department { get; set; }

        public string IsDel { get; set; }

    }
}
