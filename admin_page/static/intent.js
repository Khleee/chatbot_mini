// 이렇게 하면 첫번째 행밖에 못 가져옴...
/*
$("#examModalBtn").click(function () {
  var dataRow = $(this).closest('tr');
  console.log(dataRow.text());
  // var intent = dataRow.find('td:eq(1)').text();
  // var dscr = dataRow.find('td:eq(2)').text();
  // $("#intent").val(intent);
  // $("#intent").val(dscr);
  $("#intent").val(dataRow.find('td:eq(1)').text());
  $("#dscr").val(dataRow.find('td:eq(2)').text());
});
*/

// table의 tr 정보를 가져옴
$("#intent_list tr").click(function () {
  var tr = $(this);

  console.log(tr.text());

  var intent_no = tr.find('td:eq(0)').text();
  var intent_val = tr.find('td:eq(1)').text();
  var dscr_val = tr.find('td:eq(2)').text();

  $("#intent").val(intent_val);
  $("#dscr").val(dscr_val);
  
  $.ajax({
    type: "POST",
    url: "/example",
    dataType: "json",
    data:{
      'ino' : intent_no
    },
    error: function() {
      console.log('통신실패!');
    },
    success: function(data) {
      // console.log("통신데이터 값 : " + data);

      const select_div = document.getElementById('example_div');
      select_div.removeChild
      console.log(select_div);

      const rowLen = data.length;
      // console.log(rowLen);

      for (var i=0; i<rowLen; i++) {
        // console.log(data[i]);
        var trObj = document.createElement("tr");
        var tdObj = document.createElement("td");
        select_div.appendChild(trObj);
        trObj.appendChild(tdObj);
        tdObj.innerHTML=data[i];
      }

      console.log(select_div);
      
    }
  });
  // var db = require('config.js');
  // var mysql = require('mysql');

  // var db = mysql.createConnection({
  // host: '172.30.1.204',
  // port: 3306,
  // user: 'nlp',
  // password: 'dongwon',
  // database: 'chatbot_db'
  // });

  // db.connect();

  // db.query('SELECT * FROM intent_example', function(error, result){
  //   if(error) {
  //     console.log(error);
  //   }
  //   console.log(result);
  // });

  // db.end();
});

$("#intentAdd").click(function () {
    var intent = document.getElementById("intent");
    var dscr = document.getElementById("dscr");

    // console.log(intent.value);
    // console.log(dscr.value);
    intent.value = null;
    dscr.value = null;
    // $("#intent input[type='text']".val(""));
    // $("#dscr input[type='text']".val(""));
});

