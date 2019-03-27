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
  var start_stop='stop';
  document.getElementById("button1").onclick = function () {
      document.getElementById('button1').innerHTML = start_stop.toUpperCase(); 

      if (start_stop=='stop'){
        start_stop='start'
        console.log(start_stop)
        var image = document.getElementsByClassName("videostream");
        console.log(image)
        image.video.src = '/openpose_feed' 
      } else {
        start_stop='stop'
        var image = document.getElementsByClassName("videostream");
        image.video.src = '/vp3d_feed' 
      }
  };
};
