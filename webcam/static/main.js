var rating_data ;
var evtSource = new EventSource("/stream");

evtSource.onmessage = function(e) {
  console.log('stream')
  var classify = document.getElementById('prediction');
  var rating = document.getElementById('rating');
  var obj = JSON.parse(e.data);

  classify.innerHTML = "Exercise: " + e.action;
  rating.innerHTML = "Exercise: " + e.rating;
}

window.onload = function(){ 
  var classify = document.getElementById('prediction');
  var rating = document.getElementById('rating');
  var start_stop='stop';
  document.getElementById("button1").onclick = function () {
      document.getElementById('button1').innerHTML = start_stop.toUpperCase(); 

      if (start_stop=='stop'){
        start_stop='start'
        console.log(start_stop)
        var image = document.getElementsByClassName("videostream");
        console.log(image)
        image.video.src = '/openpose_feed'
        classify.style.display = "none";
        rating.style.display = "none";
      } else {
        start_stop='stop'
        var image = document.getElementsByClassName("videostream");

        $.ajax({
          url: "/vp3d",
          type: "get",
          data: {jsdata: ''},
          success: function(response) {
            // $("#place_for_suggestions").html(response);
              classify.style.display = "block";
              rating.style.display = "block";
              classify.innerHTML = response.prediction;
              // rating.innerHTML = response.rating;
              rating_data = response.rating;
            console.log(response.prediction);
            console.log(response.rating);
        if (rating_data!=null){
              var chart = new CanvasJS.Chart("rating", {
                animationEnabled: true,
                backgroundColor: "#333",
                title:{
                  text: "Ratings"
                },
                axisX:{
                  valueFormatString: " ",
                  crosshair: {
                    enabled: false,
                    snapToDataPoint: true
                  }
                },
                axisY: {
                  title: "",
                  gridThickness: 0,
                  includeZero: false,
                  minimum:0,
                  maximum:10,
                  // valueFormatString: "$##0.00",
                  crosshair: {
                    enabled: true,
                    snapToDataPoint: true,
                    labelFormatter: function(e) {
                      return "$" + CanvasJS.formatNumber(e.value, "##0.00");
                    }
                  }
                },
                data: [{
                  type: "area",
                  // xValueFormatString: "DD MMM",
                  // yValueFormatString: "$##0.00",
                  dataPoints:	rating_data}]
              });
              chart.render();
            }
          },
          error: function(xhr) {
            //Do Something to handle error
            console.log('error');
          }
        });  
        image.video.src = '/video_feed' 
        // location.href = '/vp3d'
        // var data_test = [
        //       { x: 1, y: 9 },
        //       { x: 2, y: 8 }
        //     ]

        // console.log(rating_data);
        // var chart = new CanvasJS.Chart("rating", {
        //   animationEnabled: true,
        //   backgroundColor: "#808080",
        //   title:{
        //     text: "Ratings"
        //   },
        //   axisX:{
        //     // valueFormatString: "DD MMM",
        //     crosshair: {
        //       enabled: true,
        //       snapToDataPoint: true
        //     }
        //   },
        //   axisY: {
        //     title: "Closing Price (in USD)",
        //     gridThickness: 0,
        //     includeZero: false,
        //     // valueFormatString: "$##0.00",
        //     crosshair: {
        //       enabled: true,
        //       snapToDataPoint: true,
        //       labelFormatter: function(e) {
        //         return "$" + CanvasJS.formatNumber(e.value, "##0.00");
        //       }
        //     }
        //   },
        //   data: [{
        //     type: "area",
        //     // xValueFormatString: "DD MMM",
        //     // yValueFormatString: "$##0.00",
        //     dataPoints:	rating_data}]
        // });
        // chart.render();
      }
  };
};
