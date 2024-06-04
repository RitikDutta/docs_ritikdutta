import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
  integrations: [
    starlight({
      title: 'Designs',
      favicon: './src/assets/me2.ico',
      head: [
        // Add ICO favicon fallback for Safari.
        {
          tag: 'link',
          attrs: {
            rel: 'icon',
            href: './src/assets/me2.ico',
            sizes: '32x32',
          },
        },
        // Add custom CSS for active navigation item
        {
          tag: 'link',
          attrs: {
            rel: 'stylesheet',
            href: '/src/assets/styles.css', // Ensure this path is correct
          },
        },
      ],

      social: {
        github: 'https://github.com/ritikdutta.com',
      },
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
          label: 'Guides',
          items: [
            // Each item here is one entry in the navigation menu.
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
