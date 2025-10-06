const DB_NAME   = "kieu_bot";
const COLL_NAME = "chunks";

// cho phép override qua biến môi trường
const INDEX_NAME = (typeof process !== "undefined" && process.env.INDEX_NAME) || "vector_index";
const DIMS = parseInt((typeof process !== "undefined" && process.env.DIMS) || "768", 10);

// kiểm tra kết nối
print(`Using DB=${DB_NAME}, coll=${COLL_NAME}, index=${INDEX_NAME}, dims=${DIMS}`);

const dbRef = db.getSiblingDB(DB_NAME);

// (tuỳ chọn) xoá index cũ nếu cần
// try { dbRef.runCommand({ dropSearchIndex: COLL_NAME, name: INDEX_NAME }); print("Dropped old index"); } catch (e) { print("No old index or drop ignored:", e.message); }

// tạo index mới
const createRes = dbRef.runCommand({
  createSearchIndexes: COLL_NAME,
  indexes: [
    {
      name: INDEX_NAME,
      definition: {
        fields: [
          // Vector field
          { path: "vector", type: "vector", numDimensions: DIMS, similarity: "cosine" },

          // Filter fields (rất quan trọng để dùng $vectorSearch + filter)
          { path: "text",            type: "filter" },
          { path: "meta.type",       type: "filter" },
          { path: "meta.source",     type: "filter" },
          { path: "meta.id",         type: "filter" },
          { path: "meta.line_range", type: "filter" }
        ]
      }
    }
  ]
});

printjson(createRes);
print(">> Done. Hãy đợi trạng thái Index = READY trong Atlas rồi hãy truy vấn.");