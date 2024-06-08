import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  integrations: [
    starlight({
      title: 'Designs',
      favicon: '/me2.ico',
      head: [
        // Add custom CSS for active navigation item
        {
          tag: 'link',
          attrs: {
            rel: 'stylesheet',
            href: '/styles.css', // Ensure this path is correct
          },
        },
        // Add Lightbox2 CSS
        {
          tag: 'link',
          attrs: {
            rel: 'stylesheet',
            href: 'https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/css/lightbox.min.css',
          },
        },
        // Add Lightbox2 JS
        {
          tag: 'script',
          attrs: {
            src: 'https://cdnjs.cloudflare.com/ajax/libs/lightbox2/2.11.3/js/lightbox.min.js',
            defer: true,
          },
        },
      ],

      social: {
        github: 'https://github.com/ritikdutta.com',
        twitter: 'https://x.com/RitikDutta7'
      },
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
            { label: 'HLD', link: '/datamigrato/hld/' },
            { label: 'LLD', link: '/datamigrato/lld/' },
          ],
        },
        {
          label: 'Data Science',
          items: [
            { label: 'Statistics', link: '/datascience/stats/' },
            { label: 'ML DL', link: '/datascience/ml_dl/' },
            { label: 'ML DL 2', link: '/datascience/ml_dl2/' },
            { label: 'Company Wise', link: '/datascience/comp_wise/' },
          ],
        },
        {
          label: 'Guides',
          items: [
            { label: 'Example Guide', link: '/guides/example/' },
          ],
        },
        {
          label: 'Reference',
          autogenerate: { directory: 'reference' },
        },
      ],
    }),
  ],
  site: 'https://docs.ritikdutta.com',
});
