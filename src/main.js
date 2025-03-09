import './style.css'
import {Tensor} from "../lib/main.js";


const t = new Tensor([[1, 2, 3], [4, 5, 6]]);
console.log(t.toString());
console.log(t.data);
console.log(t.shape);

document.querySelector('#app').innerHTML = `
  <div>
    <p>${t}</p>
  </div>
`

