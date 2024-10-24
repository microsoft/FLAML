"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[9054],{5680:(e,i,a)=>{a.d(i,{xA:()=>s,yg:()=>u});var n=a(6540);function t(e,i,a){return i in e?Object.defineProperty(e,i,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[i]=a,e}function l(e,i){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);i&&(n=n.filter((function(i){return Object.getOwnPropertyDescriptor(e,i).enumerable}))),a.push.apply(a,n)}return a}function o(e){for(var i=1;i<arguments.length;i++){var a=null!=arguments[i]?arguments[i]:{};i%2?l(Object(a),!0).forEach((function(i){t(e,i,a[i])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):l(Object(a)).forEach((function(i){Object.defineProperty(e,i,Object.getOwnPropertyDescriptor(a,i))}))}return e}function r(e,i){if(null==e)return{};var a,n,t=function(e,i){if(null==e)return{};var a,n,t={},l=Object.keys(e);for(n=0;n<l.length;n++)a=l[n],i.indexOf(a)>=0||(t[a]=e[a]);return t}(e,i);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(n=0;n<l.length;n++)a=l[n],i.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(t[a]=e[a])}return t}var p=n.createContext({}),g=function(e){var i=n.useContext(p),a=i;return e&&(a="function"==typeof e?e(i):o(o({},i),e)),a},s=function(e){var i=g(e.components);return n.createElement(p.Provider,{value:i},e.children)},y={inlineCode:"code",wrapper:function(e){var i=e.children;return n.createElement(n.Fragment,{},i)}},m=n.forwardRef((function(e,i){var a=e.components,t=e.mdxType,l=e.originalType,p=e.parentName,s=r(e,["components","mdxType","originalType","parentName"]),m=g(a),u=t,c=m["".concat(p,".").concat(u)]||m[u]||y[u]||l;return a?n.createElement(c,o(o({ref:i},s),{},{components:a})):n.createElement(c,o({ref:i},s))}));function u(e,i){var a=arguments,t=i&&i.mdxType;if("string"==typeof e||t){var l=a.length,o=new Array(l);o[0]=m;var r={};for(var p in i)hasOwnProperty.call(i,p)&&(r[p]=i[p]);r.originalType=e,r.mdxType="string"==typeof e?e:t,o[1]=r;for(var g=2;g<l;g++)o[g]=a[g];return n.createElement.apply(null,o)}return n.createElement.apply(null,a)}m.displayName="MDXCreateElement"},9026:(e,i,a)=>{a.r(i),a.d(i,{contentTitle:()=>o,default:()=>s,frontMatter:()=>l,metadata:()=>r,toc:()=>p});var n=a(8168),t=(a(6540),a(5680));const l={sidebar_label:"openai_utils",title:"autogen.oai.openai_utils"},o=void 0,r={unversionedId:"reference/autogen/oai/openai_utils",id:"reference/autogen/oai/openai_utils",isDocsHomePage:!1,title:"autogen.oai.openai_utils",description:"get\\_key",source:"@site/docs/reference/autogen/oai/openai_utils.md",sourceDirName:"reference/autogen/oai",slug:"/reference/autogen/oai/openai_utils",permalink:"/FLAML/docs/reference/autogen/oai/openai_utils",editUrl:"https://github.com/microsoft/FLAML/edit/main/website/docs/reference/autogen/oai/openai_utils.md",tags:[],version:"current",frontMatter:{sidebar_label:"openai_utils",title:"autogen.oai.openai_utils"},sidebar:"referenceSideBar",previous:{title:"completion",permalink:"/FLAML/docs/reference/autogen/oai/completion"},next:{title:"code_utils",permalink:"/FLAML/docs/reference/autogen/code_utils"}},p=[{value:"get_key",id:"get_key",children:[],level:4},{value:"get_config_list",id:"get_config_list",children:[],level:4},{value:"config_list_openai_aoai",id:"config_list_openai_aoai",children:[],level:4},{value:"config_list_from_models",id:"config_list_from_models",children:[],level:4},{value:"config_list_gpt4_gpt35",id:"config_list_gpt4_gpt35",children:[],level:4},{value:"filter_config",id:"filter_config",children:[],level:4},{value:"config_list_from_json",id:"config_list_from_json",children:[],level:4}],g={toc:p};function s(e){let{components:i,...a}=e;return(0,t.yg)("wrapper",(0,n.A)({},g,a,{components:i,mdxType:"MDXLayout"}),(0,t.yg)("h4",{id:"get_key"},"get","_","key"),(0,t.yg)("pre",null,(0,t.yg)("code",{parentName:"pre",className:"language-python"},"def get_key(config)\n")),(0,t.yg)("p",null,"Get a unique identifier of a configuration."),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Arguments"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"config")," ",(0,t.yg)("em",{parentName:"li"},"dict or list")," - A configuration.")),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Returns"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"tuple")," - A unique identifier which can be used as a key for a dict.")),(0,t.yg)("h4",{id:"get_config_list"},"get","_","config","_","list"),(0,t.yg)("pre",null,(0,t.yg)("code",{parentName:"pre",className:"language-python"},"def get_config_list(api_keys: List, api_bases: Optional[List] = None, api_type: Optional[str] = None, api_version: Optional[str] = None) -> List[Dict]\n")),(0,t.yg)("p",null,"Get a list of configs for openai api calls."),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Arguments"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"api_keys")," ",(0,t.yg)("em",{parentName:"li"},"list")," - The api keys for openai api calls."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"api_bases")," ",(0,t.yg)("em",{parentName:"li"},"list, optional")," - The api bases for openai api calls."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"api_type")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The api type for openai api calls."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"api_version")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The api version for openai api calls.")),(0,t.yg)("h4",{id:"config_list_openai_aoai"},"config","_","list","_","openai","_","aoai"),(0,t.yg)("pre",null,(0,t.yg)("code",{parentName:"pre",className:"language-python"},'def config_list_openai_aoai(key_file_path: Optional[str] = ".", openai_api_key_file: Optional[str] = "key_openai.txt", aoai_api_key_file: Optional[str] = "key_aoai.txt", aoai_api_base_file: Optional[str] = "base_aoai.txt", exclude: Optional[str] = None) -> List[Dict]\n')),(0,t.yg)("p",null,"Get a list of configs for openai + azure openai api calls."),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Arguments"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"key_file_path")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The path to the key files."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"openai_api_key_file")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The file name of the openai api key."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"aoai_api_key_file")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The file name of the azure openai api key."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"aoai_api_base_file")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The file name of the azure openai api base."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"exclude")," ",(0,t.yg)("em",{parentName:"li"},"str, optional"),' - The api type to exclude, "openai" or "aoai".')),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Returns"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"list")," - A list of configs for openai api calls.")),(0,t.yg)("h4",{id:"config_list_from_models"},"config","_","list","_","from","_","models"),(0,t.yg)("pre",null,(0,t.yg)("code",{parentName:"pre",className:"language-python"},'def config_list_from_models(key_file_path: Optional[str] = ".", openai_api_key_file: Optional[str] = "key_openai.txt", aoai_api_key_file: Optional[str] = "key_aoai.txt", aoai_api_base_file: Optional[str] = "base_aoai.txt", exclude: Optional[str] = None, model_list: Optional[list] = None) -> List[Dict]\n')),(0,t.yg)("p",null,"Get a list of configs for api calls with models in the model list."),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Arguments"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"key_file_path")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The path to the key files."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"openai_api_key_file")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The file name of the openai api key."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"aoai_api_key_file")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The file name of the azure openai api key."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"aoai_api_base_file")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The file name of the azure openai api base."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"exclude")," ",(0,t.yg)("em",{parentName:"li"},"str, optional"),' - The api type to exclude, "openai" or "aoai".'),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"model_list")," ",(0,t.yg)("em",{parentName:"li"},"list, optional")," - The model list.")),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Returns"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"list")," - A list of configs for openai api calls.")),(0,t.yg)("h4",{id:"config_list_gpt4_gpt35"},"config","_","list","_","gpt4","_","gpt35"),(0,t.yg)("pre",null,(0,t.yg)("code",{parentName:"pre",className:"language-python"},'def config_list_gpt4_gpt35(key_file_path: Optional[str] = ".", openai_api_key_file: Optional[str] = "key_openai.txt", aoai_api_key_file: Optional[str] = "key_aoai.txt", aoai_api_base_file: Optional[str] = "base_aoai.txt", exclude: Optional[str] = None) -> List[Dict]\n')),(0,t.yg)("p",null,"Get a list of configs for gpt-4 followed by gpt-3.5 api calls."),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Arguments"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"key_file_path")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The path to the key files."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"openai_api_key_file")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The file name of the openai api key."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"aoai_api_key_file")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The file name of the azure openai api key."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"aoai_api_base_file")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The file name of the azure openai api base."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"exclude")," ",(0,t.yg)("em",{parentName:"li"},"str, optional"),' - The api type to exclude, "openai" or "aoai".')),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Returns"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"list")," - A list of configs for openai api calls.")),(0,t.yg)("h4",{id:"filter_config"},"filter","_","config"),(0,t.yg)("pre",null,(0,t.yg)("code",{parentName:"pre",className:"language-python"},"def filter_config(config_list, filter_dict)\n")),(0,t.yg)("p",null,"Filter the config list by provider and model."),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Arguments"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"config_list")," ",(0,t.yg)("em",{parentName:"li"},"list")," - The config list."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"filter_dict")," ",(0,t.yg)("em",{parentName:"li"},"dict, optional")," - The filter dict with keys corresponding to a field in each config,\nand values corresponding to lists of acceptable values for each key.")),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Returns"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"list")," - The filtered config list.")),(0,t.yg)("h4",{id:"config_list_from_json"},"config","_","list","_","from","_","json"),(0,t.yg)("pre",null,(0,t.yg)("code",{parentName:"pre",className:"language-python"},'def config_list_from_json(env_or_file: str, file_location: Optional[str] = "", filter_dict: Optional[Dict[str, Union[List[Union[str, None]], Set[Union[str, None]]]]] = None) -> List[Dict]\n')),(0,t.yg)("p",null,"Get a list of configs from a json parsed from an env variable or a file."),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Arguments"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"env_or_file")," ",(0,t.yg)("em",{parentName:"li"},"str")," - The env variable name or file name."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"file_location")," ",(0,t.yg)("em",{parentName:"li"},"str, optional")," - The file location."),(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"filter_dict")," ",(0,t.yg)("em",{parentName:"li"},"dict, optional")," - The filter dict with keys corresponding to a field in each config,\nand values corresponding to lists of acceptable values for each key.\ne.g.,")),(0,t.yg)("pre",null,(0,t.yg)("code",{parentName:"pre",className:"language-python"},'filter_dict = {\n    "api_type": ["open_ai", None],  # None means a missing key is acceptable\n    "model": ["gpt-3.5-turbo", "gpt-4"],\n}\n')),(0,t.yg)("p",null,(0,t.yg)("strong",{parentName:"p"},"Returns"),":"),(0,t.yg)("ul",null,(0,t.yg)("li",{parentName:"ul"},(0,t.yg)("inlineCode",{parentName:"li"},"list")," - A list of configs for openai api calls.")))}s.isMDXComponent=!0}}]);