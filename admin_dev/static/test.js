function getDialogId() {
    let dialogList = document.getElementById('dialog_list');

    for (let i = 1; i < dialogList.rows.length; i++) {
        dialogList.rows[i].cells[7].onclick = function () {
          let dialog_id = dialogList.rows[i].cells[0].innerText;
          alert(dialog_id+"번 dialog를 선택하셨습니다.");
       }
    }
}

function addRow(el) {
    // table element 찾기
    console.log(el);
    console.log(el.value);
    // const table = document.getElementById('dialog_list');
    
    // // 새 행(Row) 추가 (테이블 중간에)
    // const newRow = table.insertRow(1);
    
    // // 새 행(Row)에 Cell 추가
    // const newCell1 = newRow.insertCell(0);
    // const newCell2 = newRow.insertCell(1);
    
    // // Cell에 텍스트 추가
    // newCell1.innerText = '새 과일';
    // newCell2.innerText = 'New Fruit';
}

// $("#addrow").click(function(){
//     var checkBtn = $(this);

//     var tr = checkBtn.parent().parent();
//     console.log(tr);

//     var idx = tr.text();

//     console.log(idx);
// })

// 테이블의 Row 클릭시 값 가져오기
$("#dialog_list tr").click(function(){ 	

    var str = ""
    var tdArr = new Array();	// 배열 선언
    
    // 현재 클릭된 Row(<tr>)
    var tr = $(this);
    var td = tr.children();
    
    // tr.text()는 클릭된 Row 즉 tr에 있는 모든 값을 가져온다.
    console.log("클릭한 Row의 모든 데이터 : "+tr.text());
})