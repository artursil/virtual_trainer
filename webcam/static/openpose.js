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
      location.href = "/";
  };
};
