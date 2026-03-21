import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMermaid from 'astro-mermaidjs/remark-mermaid';

export default defineConfig({
  integrations: [
    starlight({
      title: 'Designs',
      favicon: '/me2.ico',
      head: [
        {
          tag: 'link',
          attrs: {
            rel: 'stylesheet',
            href: '/styles.css',
          },
        },
      ],
      social: [
        {
          label: 'GitHub',
          href: 'https://github.com/ritikdutta.com',
          icon: 'github',
        },
        {
          label: 'Twitter',
          href: 'https://x.com/RitikDutta7',
          icon: 'twitter',
        },
      ],
      sidebar: [
        {
          label: 'CWEM Docs',
          items: [
            { label: 'CWEM Guide', link: '/guides/company_work_environment_management/' },
            { label: 'HLD', link: '/guides/hld/' },
            { label: 'LLD', link: '/guides/lld/' },
            { label: 'Architecture', link: '/guides/architecture/' },
            { label: 'Wireframe', link: '/guides/wireframe/' },
            { label: 'KPI', link: '/guides/kpi/' },
          ],
        },
        {
          label: 'Datamigrato Docs',
          items: [
            { label: 'Datamigrato Guide', link: '/datamigrato/datamigrato_guide/' },
            { label: 'HLD', link: '/datamigrato/hld/' },
            { label: 'LLD', link: '/datamigrato/lld/' },
          ],
        },
        {
          label: 'Interview Ready',
          items: [
            { label: 'Introduction', link: '/interview_ready/intro/' },
            { label: 'HLD', link: '/interview_ready/hld/' },
            { label: 'LLD', link: '/interview_ready/lld/' },
          ],
        },
      ],
    }),
  ],
  markdown: {
    syntaxHighlight: {
      type: 'shiki',
      excludeLangs: ['mermaid'],
    },
    remarkPlugins: [remarkMermaid],
  },
  site: 'https://docs.ritikdutta.com',
});