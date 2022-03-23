// @ts-check

const map = document.getElementById("main_canvas");
const mapContext = map.getContext("2d");

const img = document.getElementById("main_image");

img.addEventListener("load", () => mapContext.drawImage(img, 0, 0, img.width, img.height, 0, 0, map.width, map.height));

requestAnimationFrame(drawCanvas); 

const view = (() => {
  let m = [1, 0, 0, 1, 0, 0]; // current view transform
  let scale = 1;              // current scale
  var mapContext;             // reference to the 2D context
  const pos = { x: 0, y: 0 }; // current position of origin
  var dirty = true;

  const API = {
    set context(_mapContext) { mapContext = _mapContext; dirty = true },
    apply() {
      if (dirty) { this.update() }
      mapContext.setTransform(m[0], m[1], m[2], m[3], m[4], m[5])
    },
    get scale() { return scale },
    get position() { return pos },
    isDirty() { return dirty },
    update() {
      dirty = false;
      m[3] = m[0] = scale;
      m[2] = m[1] = 0;
      m[4] = pos.x;
      m[5] = pos.y;
    },
    pan(amount) {
      if (dirty) { this.update() }
       pos.x += amount.x;
       pos.y += amount.y;
       dirty = true;
    },
    scaleAt(at, amount) { // at in screen coords
      if (dirty) { this.update() }
      scale *= amount;
      pos.x = at.x - (at.x - pos.x) * amount;
      pos.y = at.y - (at.y - pos.y) * amount;
      dirty = true;
    },
    toWorld(x, y) {  // convert to world coordinates
        x = (x - pos.x) / scale;
        y = (y - pos.y) / scale;
        return {x, y};
    }
  };
  return API;
})();

view.context = mapContext;

function drawCanvas() {
    if (view.isDirty()) { 
        mapContext.setTransform(1, 0, 0, 1, 0, 0); 
        mapContext.clearRect(0, 0, map.width, map.height);

        view.apply(); // set the 2D context transform to the view
        mapContext.drawImage(img, 0, 0, img.width, img.height, 0, 0, map.width, map.height);
    }
    requestAnimationFrame(drawCanvas);
}


map.addEventListener("mousemove", mouseEvent, {passive: true});
map.addEventListener("mousedown", mouseEvent, {passive: true});
map.addEventListener("mouseup", mouseEvent, {passive: true});
map.addEventListener("mouseout", mouseEvent, {passive: true});
map.addEventListener("wheel", mouseWheelEvent, {passive: false});

const mouse = {x: 0, y: 0, oldX: 0, oldY: 0, button: false};

function mouseEvent(event) {
    if (event.type === "mousedown") { 
        mouse.button = true
        showDetails(event); 
    }
    if (event.type === "mouseup" || event.type === "mouseout") { mouse.button = false }
    mouse.oldX = mouse.x;
    mouse.oldY = mouse.y;
    mouse.x = event.offsetX;
    mouse.y = event.offsetY    
    if(mouse.button) { // pan
        view.pan({x: mouse.x - mouse.oldX, y: mouse.y - mouse.oldY});
    }
}

function mouseWheelEvent(event) {
    var x = event.offsetX;
    var y = event.offsetY;
    if (event.deltaY < 0) { view.scaleAt({x, y}, 1.1) }
    else { view.scaleAt({x, y}, 1 / 1.1) }
    event.preventDefault();
}

function showDetails({x, y})
{
    console.log(view.toWorld(x, y));
}