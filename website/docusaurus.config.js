/** @type {import('@docusaurus/types').DocusaurusConfig} */
const math = require('remark-math');
const katex = require('rehype-katex');

module.exports = {
  title: 'FLAML',
  tagline: 'A Fast Library for Automated Machine Learning & Tuning',
  url: 'https://shao-kun-zhang.github.io/',
  baseUrl: '/FLAML/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'Shao-kun-Zhang', // Usually your GitHub org/user name.
  projectName: 'FLAML', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: 'FLAML',
      logo: {
        alt: 'FLAML',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'doc',
          docId: 'Getting-Started',
          position: 'left',
          label: 'Docs',
        },
        {
            type: 'doc',
            docId: 'reference/automl',
            position: 'left',
            label: 'SDK',
        },
        {
          type: 'doc',
          docId: 'FAQ',
          position: 'left',
          label: 'FAQ',
        },
        {
          href: 'https://github.com/microsoft/FLAML',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        // {
        //   title: 'Docs',
        //   items: [
        //     {
        //       label: 'Getting Started',
        //       to: 'docs/getting-started',
        //     },
        //   ],
        // },
        // {
        //   title: 'Community',
        //   items: [
        //     // {
        //     //   label: 'Stack Overflow',
        //     //   href: 'https://stackoverflow.com/questions/tagged/pymarlin',
        //     // },
        //     // {
        //     //   label: 'Discord',
        //     //   href: 'https://discordapp.com/invite/docusaurus',
        //     // },
        //     // {
        //     //   label: 'Twitter',
        //     //   href: 'https://twitter.com/docusaurus',
        //     // },
        //   ],
        // },
        // {
        //   title: 'More',
        //   items: [
        //     {
        //       label: 'GitHub',
        //       href: 'https://github.com/microsoft/FLAML',
        //     },
        //   ],
        // },
      ],
      // copyright: `Copyright © ${new Date().getFullYear()} Microsoft Inc. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/microsoft/FLAML/edit/master/website/',
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        // blog: {
        //   showReadingTime: true,
        //   // Please change this to your repo.
        //   editUrl:
        //     'https://github.com/facebook/docusaurus/edit/master/website/blog/',
        // },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  stylesheets: [
    {
        href: "https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.css",
        integrity: "sha384-Um5gpz1odJg5Z4HAmzPtgZKdTBHZdw8S29IecapCSB31ligYPhHQZMIlWLYQGVoc",
        crossorigin: "anonymous",
    },
  ],  
};
