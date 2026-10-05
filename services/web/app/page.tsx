import { NewStudyForm } from "@/components/new-study-form"
import { RunsTable } from "@/components/runs-table"

export default function Page() {
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
      <NewStudyForm />
      <RunsTable />
    </div>
  )
}
