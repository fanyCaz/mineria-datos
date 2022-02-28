let status = [1,2,3,4,5,6,7];

let activities = ["register","examine","check","decide"];

let t = [{s: 1, a: "inicio", e: 2},
         {s: 2, a: "register", e: 3}
        ];

let transition = {};
status.forEach((state) => {
  let template = document.querySelector('.status-template').cloneNode(true);
  template.id = state;
  document.querySelector('.diagram-wrapper').append(template);
  console.log(state);
});

status.forEach((state) => {
  transition = t.filter((s)=> s.s == state)[0];
  console.log( transition); 
  if(transition !== undefined){
    document.getElementById(transition.s).append("<p>criminal</p>");
  }

});
