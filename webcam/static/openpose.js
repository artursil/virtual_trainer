var evtSource = new EventSource("/stream");

evtSource.onmessage = function(e) {
  var classify = document.getElementById('classify');
  var rating = document.getElementById('rating');
  var obj = JSON.parse(e.data);

  classify.innerHTML = "Exercise: " + e.action;
  rating.innerHTML = "Exercise: " + e.rating;
}

window.onload = function(){ 
  document.getElementById("button1").onclick = function () {
<<<<<<< HEAD
      location.href = "/vp3d";
=======
      location.href = "/";
>>>>>>> 7c3f33bd744767a8a7eb4eb5a01d7b1b1cae8f8f
  };
};
