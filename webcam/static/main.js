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
  var welcome = document.getElementById('welcome');
  var start_stop='stop';
  welcome.style.display="block";
  classify.style.display = "none";
  document.getElementById("button1").onclick = function () {
      document.getElementById('button1').innerHTML = start_stop.toUpperCase(); 

      if (start_stop=='stop'){
        start_stop='start';
        console.log(start_stop);
        var image = document.getElementsByClassName("videostream");
        console.log(image);
        welcome.style.display="block";
        image.video.src = '/openpose_feed';
        classify.style.display = "none";
        rating.style.display = "none";
      } else {
        start_stop='stop';
        var image = document.getElementsByClassName("videostream");

        $.ajax({
          url: "/vp3d",
          type: "get",
          data: {jsdata: ''},
          success: function(response) {
            // $("#place_for_suggestions").html(response);
              console.log(response.rating);
              if (response.rating==null){
                classify.style.display = "block";
              }
              rating.style.display = "block";
              welcome.style.display="none";
              classify.innerHTML = response.prediction;
              // rating.innerHTML = response.rating;
              rating_data = response.rating;
            console.log(response.prediction);
            console.log(response.rating);
        if (rating_data!=null){
              var chart = new CanvasJS.Chart("rating", {
                animationEnabled: true,
                backgroundColor: 'transparent',
                
                title:{
                  text: response.prediction.charAt(0).toUpperCase() + response.prediction.slice(1)
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
                  color: "#ff7f00",
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
      }
  };
};
