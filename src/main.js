import './style.css'
import {Tensor} from "../lib/main.js";


const t = Tensor.randn([2, 3]);
console.log(t.toString());
console.log(t.data);
console.log(t.shape);

document.querySelector('#app').innerHTML = `
  <div>
    <p>${t}</p>
  </div>
`

