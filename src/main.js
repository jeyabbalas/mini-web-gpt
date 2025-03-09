import './style.css'
import {test} from "../lib/main.js";

document.querySelector('#app').innerHTML = `
  <div>
    <h1>${test()}</h1>
  </div>
`

