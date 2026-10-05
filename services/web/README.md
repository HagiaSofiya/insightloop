# Next.js template

This is a Next.js template with shadcn/ui.

## Adding components

To add components to your app, run the following command:

```bash
npx shadcn@latest add button
```

This will place the ui components in the `components` directory.

## Using components

To use the components in your app, import them as follows:

```tsx
import { Button } from "@/components/ui/button";
```

## Insightloop notes

- `make web` (from the repo root) starts this app on port 3100 (the API's CORS list allows 3000 and 3100). Set `NEXT_PUBLIC_API_URL` if the API is not on `http://localhost:8000`.
- The API must allow this origin (see `CORSMiddleware` in `services/api/app/main.py`) and a worker must be running (`make worker`) or runs never advance.
