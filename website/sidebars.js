/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

 module.exports = {
  docsSidebar: [
    'Getting-Started', 
    'Installation',
    {'Use Cases': [{type: 'autogenerated', dirName: 'Use Cases'}]},
    {'Examples': [{type: 'autogenerated', dirName: 'Examples'}]},
    'Develop',
    'Research',
  ],
  // pydoc-markdown auto-generated markdowns from docstrings
  referenceSideBar: [require("./docs/reference/sidebar.json")]
};
