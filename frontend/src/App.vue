<!-- 根 Vue 组件：交给 router-view 渲染当前页面，并定义全站基础字体、视口和控件样式。
     三栏滚动规则由聊天页及子组件负责，避免根容器干扰其独立纵向滚动。 -->

<template>
<!--  当前 URL 对应哪个页面，Router 就把哪个页面放进这里。-->
  <router-view />
</template>

<style>
:root { color: #1f2937; background: #f7f9fc; font-family: "Microsoft YaHei", system-ui, sans-serif; }
* { box-sizing: border-box; }
html, body, #app { height: 100%; overflow: hidden; }
body { margin: 0; }
button, input, textarea { font: inherit; }
button { cursor: pointer; }
</style>
