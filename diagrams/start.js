let status = [1,2,3,4,5,6,7,8];

let activities = ["register","examine","check","decide"];

let t = [{s: 1, a: "inicio", e: 2},
         {s: 2, a: "register request", e: 3},
         {s: 3, a: "examine casually", e: 4},
         {s: 3, a: "examine casually", e: 5},
         {s: 4, a: "examine casually", e: 3},
         {s: 4, a: "check ticket", e: 5},
         {s: 5, a: "decide", e: 6},
         {s: 5, a: "decide", e: 7},
         {s: 6, a: "reinitiate request", e: 4},
         {s: 6, a: "reinitiate request", e: 3},
         {s: 7, a: "reject request", e: 8}
        ];

let transition = {};
status.forEach((state) => {
  let template = document.querySelector('.status-template').cloneNode(true);
  template.id = state;
  template.innerText = `S${state}`
  document.querySelector('.diagram-wrapper').append(template);
  console.log(state);
});

status.forEach((state) => {
  transition = t.filter((s)=> s.s == state)[0];

  let lineConector = document.querySelector('.connect-line').cloneNode(true);
  if(transition !== undefined){
    lineConector = lineConector.querySelector('line');
    let stateDiv = document.getElementById(transition.s);
    let stateEndDiv = document.getElementById(transition.e);
    //with offsetLeft,offsetHeight are the coordinates as x,y of an element

    lineConector.attributes.x1.value = stateDiv.offsetWidth;
    lineConector.attributes.x2.value = stateDiv.offsetWidth;
    lineConector.attributes.y1.value = stateDiv.offsetTop;
    lineConector.attributes.y2.value = stateEndDiv.offsetTop;

    //connect-line
  }

});
