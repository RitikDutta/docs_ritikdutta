import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
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
            // Each item here is one entry in the navigation menu.
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
            // Each item here is one entry in the navigation menu.
            { label: 'Datamigrato Guide', link: '/datamigrato/datamigrato_guide/' },
            { label: 'HLD', link: '/datamigrato/hld/' },
            { label: 'LLD', link: '/datamigrato/lld/' },
          ],
        },
        {
          label: 'Interview Ready',
          items: [
            { label: 'Introduction', link: '/interview_ready/intro/' },
          ],
        },
        // {
        //   label: 'Data Science',
        //   items: [
        //     // Each item here is one entry in the navigation menu.
        //     { label: 'Statistics', link: '/datascience/stats/' },
        //     { label: 'ML DL', link: '/datascience/ml_dl/' },
        //     { label: 'ML DL 2', link: '/datascience/ml_dl2/' },
        //     { label: 'Company Wise', link: '/datascience/comp_wise/' },
        //   ],
        // },
        // {
        //   label: 'Guides',
        //   items: [
        //     // Each item here is one entry in the navigation menu.
        //     { label: 'Example Guide', link: '/guides/example/' },
        //   ],
        // },
        // {
        //   label: 'Reference',
        //   autogenerate: { directory: 'reference' },
        // },
      ],
    }),
  ],
  site: 'https://docs.ritikdutta.com',
});
