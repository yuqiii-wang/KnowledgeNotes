# DAO (Data Access Object)

Data Access Object (DAO) pattern is a structural pattern that separates the data persistence logic from the business logic.
It is used inside "service" to fulfill business logic flow.

Persistence is a means to persist and retrieve data/information from a non-volatile storage system, e.g., disk.

ORM (Object-Relational Mapping) is about how to map java object to a SQL DB row.

## DAO Pattern with JPA/Hibernate

JPA/Hibernate is spring default DAO implementation.

JPA (Java Persistence API) represents a simplification of the persistence programming model, explicitly defined object-relational mapping rules.

Hibernate ORM (or simply Hibernate) is an object-relational mapping tool that satisfies JPA specifications.

For example, below SQL table can be represented by a Java bean.

```sql
create table products (
    id bigint not null auto_increment,
    product_name varchar(255) not null,
    product_price double not null,
    primary key (id)
);
```

Below is the Java bean `Product`.
The annotated `@Entity` is used by Hibernate to automatically generate SQL for later use by `@Repository`.

```java
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Column;
import javax.persistence.Table;

@Entity
@Table(name = "products")
public class Product {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "product_name", nullable = false)
    private String name;

    @Column(name = "product_price", nullable = false)
    private double price;

    // Getters and setters
    // ...
}
```

JPA/Hibernate uses `@Repository` that `extends JpaRepository<Product, Long>` to implement CRUD.

```java
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ProductRepository extends JpaRepository<Product, Long> {
    // Additional query methods can be defined here
    List<Product> findByName(String name);

    Product findById(long id);
}
```

DAO is used by service.

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ProductService {

    @Autowired
    private ProductRepository productRepository;

    public List<Product> findByName() {
        return productRepository.findByName();
    }

    public void findById(Long id) {
        productRepository.findById(id);
    }
}
```

## DAO Pattern with MyBatis

MyBatis is an alternative DAO solution.

Unlike JPA/Hibernate, MyBatis focuses on mapping SQL queries to Java methods and does not enforce ORM (Object-Relational Mapping).

MyBatis uses `@Mapper` directly executing SQL to query DB.

Given the implementation of directly executing SQL, the pros and cons are

* Easy, flexible and fast compared to JPA/Hibernate
* Need to manually write SQL to map object vs SQL

```java
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface ProductMapper {

    @Select("SELECT * FROM products")
    List<Product> findAll();

    @Select("SELECT * FROM products WHERE id = #{id}")
    Product findById(Long id);

    @Insert("INSERT INTO products(product_name, product_price) VALUES(#{name}, #{price})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    void insert(Product product);

    @Update("UPDATE products SET product_name=#{name}, product_price=#{price} WHERE id = #{id}")
    void update(Product product);

    @Delete("DELETE FROM products WHERE id = #{id}")
    void delete(Long id);
}
```

The `@Mapper` annotated `ProductMapper` is directly used in service.

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ProductService {

    @Autowired
    private ProductMapper productMapper;

    public List<Product> findAllProducts() {
        return productMapper.findAll();
    }

    public void saveProduct(Product product) {
        if (product.getId() == null) {
            productMapper.insert(product);
        } else {
            productMapper.update(product);
        }
    }

    public void deleteProduct(Long id) {
        productMapper.delete(id);
    }
}
```
