# Spring MVC Design

MVC (Model-View-Controller) is an architectural pattern how to interact with end user from UI/*view*, passing HTTP requests to backend controlled/managed by *controller*, then such requests are further managed by *model* that perform CRUD operations on database.

## Prerequisites

### Beans

Recall that a *bean* refers to an object that is managed by the Spring IoC, i.e., create an object via config.

In `Product.java`, define a class `Product`.

```java
public class Product {
    private Long id;
    private String productName;
    private Integer productPrice;

    // getter and setter
    public long getId() {return id;}
    public String getProductName() {return productName;}
    public Integer getProductPrice() {return productPrice;}
    public long setId(long id) { this.id = id;}
    public String setProductName(String productName) { this.id = productName;}
    public Integer setProductPrice(Integer productPrice) { this.productPrice = productPrice;}
}
```

In `AppConfig.java`, where `@Configuration` and `@Bean` annotations are used to define beans.

```java
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class AppConfig {

    @Bean
    public Product Product() {
        Product productBean = new Product();
        productBean.setProductName("XiaoMi Su 7");
        productBean.setProductPrice(210000);
        return productBean;
    }
}
```

### MVC (Model-View-Controller) In Java Spring

#### Model

Contained business logic and data models.

For example, the `Product.java` can be replaced with the below.

```java
@Entity
public class Product {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String productName;
    private Integer productPrice;
    
    // getters and setters
}

@Repository
public interface ProductRepository extends JpaRepository<Product, Long> {
}

@Service
public class ProductService {
    @Autowired
    private ProductRepository productRepository;

    public List<Product> findAllProducts() {
        return productRepository.findAll();
    }

    public Product saveProduct(Product product) {
        return productRepository.save(product);
    }
}
```

#### View

How variables are displayed on html.

```html
<!DOCTYPE html>
<html xmlns:th="http://www.thymeleaf.org">
<head>
    <title>Product List</title>
</head>
<body>
    <h1>Products</h1>
    <ul>
        <li th:each="product : ${products}">
            <span th:text="${product.name}">Product Name</span> - 
            <span th:text="${product.price}">Product Price</span>
        </li>
    </ul>
</body>
</html>
```

#### Controller

Interact between view and model, take HTTP requests from view/UI and parse as CRUD operation to backend model.

```java
@Controller
@RequestMapping("/products")
public class ProductController {
    
    @Autowired
    private ProductService productService;
    
    @GetMapping
    public String listProducts(Model model) {
        List<Product> products = productService.findAllProducts();
        model.addAttribute("products", products);
        return "productList";  // View name (productList.html)
    }
    
    @PostMapping
    public String addProduct(Product product) {
        productService.saveProduct(product);
        return "redirect:/products";  // Redirect to the list view
    }
}
```

## Annotations in Spring for MVC

* `@Component`

`@Component` is used across the spring framework to mark beans as Spring's managed generic components, and many specialist components, e.g., `@Repository`, `@Service` and `@Controller` are inherited from `@Component` with added specialist functions.
The advantage of `@Component` compared to a typical java bean is that it can utilize rich spring framework functions.

For example, lifecycle management by spring,

```java
@Component
public class MyComponent {
    @PostConstruct
    public void init() { ... // Initialization code
    }

    @PreDestroy
    public void cleanup() { ... // Cleanup code
    }
}
```

Event handling and listening,

```java
@Component
public class MyEventHandler {
    @EventListener
    public void handleContextRefresh(ContextRefreshedEvent event) {
        System.out.println("Context Refreshed Event received.");
    }

    @Scheduled(fixedRate = 5000)
    public void performTask() {
        System.out.println("Scheduled task running every 5 seconds");
    }
}
```

* `Entity`

`@Entity`: Relevant in the persistence layer, particularly with ORM (Object-Relational Mapping) frameworks like Hibernate. It interacts with the database, representing data and managing CRUD operations.

```java
@Entity
public class Product {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String productName;
    private Integer productPrice;
    
    // getters and setters
    ...
}
```

ORM (Object-Relational Mapping) is about how to map java object to a SQL DB row.
For example, `class Product` can be mapped as a row in this SQL table `products`.

```sql
create table products (
    id bigint not null auto_increment,
    product_name varchar(255) not null,
    product_price double not null,
    primary key (id)
);
```

* `@Repository`

```java
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface ProductRepository extends JpaRepository<Product, Long> {
    // Custom query methods (if needed) can be added here
}
```

* `@Service`

A `@Service` is considered an intermediary between controller `@Controller` and model `@Repository`.

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class ProductService {

    @Autowired
    private ProductRepository productRepository;

    public List<Product> findAllProducts() {
        return productRepository.findAll();
    }

    public Product saveProduct(Product product) {
        return productRepository.save(product);
    }

    public void deleteProduct(Long id) {
        productRepository.deleteById(id);
    }
}
```

For large and complex project, services are used to implement complex business logics.

For example, a product dealer system might have below pseudo-code services,
where `ProductDealerProcessor` serves as a centralized service processor, reading messages from various sources and querying Mongo DB and product manufacturer for info (handled by various responsible services), then gets back to customers.

```java
@Component
public class ProductDealerProcessor implements Processor {
    @Autowired
    private FrontOfficeChatMessageService frontOfficeChatMessageService;
    @Autowired
    private OnlineChatMessageService onlineChatMessageService;
    @Autowired
    private CacheMessageService cacheMessageService;
    @Autowired
    private MongoInventoryService mongoInventoryService;
    @Autowired
    private ProductManufacturerChatService productManufacturerChatService;

    @Override
    public OutMessage process(ContextInfo contextInfo, InMessage inMessage) throws Exception {
        ... // business logics about how above auto-wired services are processed
            // for example...
        
        cacheMessageService.cache_msg(inMessage);
        CustomerQuery customerQuery;
        if (contextInfo.msg_type() == "FrontOffice") {
            customerQuery = frontOfficeChatMessageService.parseMsg(inMessage);
        }
        else if (contextInfo.msg_type() == "OnlineChat") {
            customerQuery = onlineChatMessageService.parseMsg(inMessage);
        }
        
        ProductInventoryInfo productInventoryInfo = mongoInventoryService.ask(customerQuery);
        ProductInfo productInfo = productManufacturerChatService.ask(customerQuery);

        OutMessage outMsg(productInventoryInfo, productInfo);

        return outMsg;
    }
}
```

* `@Controller`

`@Controller` is used to receive HTTP requests and invokes corresponding services.

```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/products")
public class ProductController {

    @Autowired
    private ProductService productService;

    @GetMapping
    public List<Product> getAllProducts() {
        return productService.findAllProducts();
    }

    @PostMapping
    public Product createProduct(@RequestBody Product product) {
        return productService.saveProduct(product);
    }

    @DeleteMapping("/{id}")
    public void deleteProduct(@PathVariable Long id) {
        productService.deleteProduct(id);
    }
}
```