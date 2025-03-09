import './style.css'
import {Tensor} from "../lib/main.js";


const t = new Tensor([3, 3], true);
console.log(t.toString());

document.querySelector('#app').innerHTML = `
  <div>
    <p>${t}</p>
  </div>
`

